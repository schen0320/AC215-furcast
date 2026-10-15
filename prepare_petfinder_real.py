"""Audited PetFinder benchmark preparation; never equates AdoptionSpeed and LOS.

Requires a real official extraction and completed image audit. Raw tables remain
unchanged. Grouping includes excluded/multi-pet and unlabeled bridge records.
Only labeled eligible records enter the internal group holdouts or fit stage.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from furcast_data import PF_REQUIRED, PF_CAT, PF_NUM, connected_groups, verify_splits

# Fields plausibly changed during a listing are excluded by default. Even these
# remaining fields do not have independently verified first-listing timestamps.
CAT=['Type','Breed1','Breed2','Gender','Color1','Color2','Color3',
     'MaturitySize','FurLength','State']
NUM=['Age']
FEATURES=CAT+NUM
OMITTED=['AdoptionSpeed','Name','PetID','RescuerID','Quantity','Description',
         'PhotoAmt','VideoAmt','Vaccinated','Dewormed','Sterilized','Health','Fee']
OUTCOME_PATTERN=r'(?i)\b(?:adopted|rehomed|re-homed|found (?:a |their )?(?:new |forever )?home|no longer available)\b|已[被获]?领养|已送养|已找到新家'
GENERIC_PATTERN=r'(?i)mixed breed|domestic|tabby|calico|tortoiseshell|torbie|tuxedo|dilute|^terrier$'


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')


def validate_table(frame,labeled):
    required=set(PF_REQUIRED)-({'AdoptionSpeed'} if not labeled else set())
    if not required.issubset(frame):raise ValueError('Missing columns: '+str(required-set(frame)))
    if frame.PetID.isna().any() or frame.PetID.duplicated().any():raise ValueError('Invalid PetID')
    for c in PF_CAT+PF_NUM+['Quantity','PhotoAmt','VideoAmt']+(['AdoptionSpeed'] if labeled else []):
        v=pd.to_numeric(frame[c],errors='raise')
        if v.isna().any() or (v<0).any() or (v%1!=0).any():raise ValueError('Invalid integer '+c)
    enums={'Type':{1,2},'Gender':{1,2,3},'MaturitySize':{0,1,2,3,4},
           'FurLength':{0,1,2,3},'Vaccinated':{1,2,3},'Dewormed':{1,2,3},
           'Sterilized':{1,2,3},'Health':{0,1,2,3},'AdoptionSpeed':set(range(5))}
    for c,allowed in enums.items():
        if c in frame and not frame[c].isin(allowed).all():raise ValueError('Invalid enum '+c)
    if not frame.Quantity.ge(1).all():raise ValueError('Quantity must be positive')


def audit_json_files(raw,table,images,out):
    rows=[]
    known=set(table.PetID); image_names=set(images.source_file.str.split('/').str[-1].str.replace('.jpg','',regex=False))
    for part in ['train_metadata','test_metadata','train_sentiment','test_sentiment']:
        keys=Counter(); malformed=[]; orphan=[]; count=0
        files=sorted((raw/part).glob('*.json'))
        for path in files:
            count+=1
            try:
                value=json.loads(path.read_text())
                if not isinstance(value,dict):raise ValueError('Expected JSON object')
                keys.update(value.keys())
            except (ValueError,UnicodeError):malformed.append(path.name)
            if part.endswith('metadata'):
                if path.stem not in image_names:orphan.append(path.name)
            elif path.stem not in known:orphan.append(path.name)
        rows.append({'directory':part,'files':count,'malformed':malformed,'orphan_files':orphan,'top_level_key_counts':dict(keys)})
    json_write(out/'optional_json_audit.json',rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True)
    p.add_argument('--images',required=True,help='image_manifest.csv from audit_petfinder_images.py')
    p.add_argument('--edges',required=True,help='petfinder_image_candidates.csv')
    p.add_argument('--out',required=True);p.add_argument('--export-images',required=True)
    a=p.parse_args();raw=Path(a.raw);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    train=pd.read_csv(raw/'train/train.csv',dtype={'PetID':str,'RescuerID':str})
    test=pd.read_csv(raw/'test/test.csv',dtype={'PetID':str,'RescuerID':str})
    validate_table(train,True);validate_table(test,False)
    if 'AdoptionSpeed' in test:raise ValueError('Unexpected labeled official test; inspect source version')
    if set(train.PetID)&set(test.PetID):raise ValueError('Overlapping official PetIDs')
    train['source_partition']='train';test['source_partition']='test'
    table=pd.concat([train,test],ignore_index=True)
    lookup=pd.read_csv(raw/'breed_labels.csv');breed=lookup.set_index('BreedID')
    if not breed.index.is_unique:raise ValueError('Ambiguous breed dictionary')
    table['schema_review_reason']=''
    for c in ['Breed1','Breed2']:
        unknown=table[c].ne(0)&~table[c].isin(breed.index)
        mismatch=table[c].ne(0)&table[c].map(breed.Type).ne(table.Type)
        table.loc[unknown,'schema_review_reason']+=c+'_unknown;'
        table.loc[mismatch&~unknown,'schema_review_reason']+=c+'_species_mismatch;'
    for c in ['Color1','Color2','Color3']:
        if not table[c].isin(set(pd.read_csv(raw/'color_labels.csv').ColorID)|{0}).all():raise ValueError('Unknown color code')
    if not table.State.isin(pd.read_csv(raw/'state_labels.csv').StateID).all():raise ValueError('Unknown State')
    table.loc[table.Quantity.eq(1)&table.Gender.eq(3),'schema_review_reason']+='single_pet_mixed_gender;'
    table['primary_breed_name']=table.Breed1.map(breed.BreedName)
    table['secondary_breed_name']=table.Breed2.map(breed.BreedName)
    table['outcome_text_review']=table.Description.fillna('').str.contains(OUTCOME_PATTERN,regex=True)
    images=pd.read_csv(a.images,dtype={'phash':str});edges=pd.read_csv(a.edges)
    if not set(images.PetID).issubset(set(table.PetID)):raise ValueError('Orphan image PetID')
    if images.image_key.duplicated().any():raise ValueError('Duplicate image keys')
    partition=table.set_index('PetID').source_partition
    if images.PetID.map(partition).ne(images.source_partition).any():raise ValueError('Wrong official image partition')
    good=images.loc[images.decode_ok]
    counts=good.groupby('PetID').size()
    default=good.loc[good.image_number.eq(1)].set_index('PetID')
    if not default.index.is_unique:raise ValueError('Ambiguous default photo')
    table['actual_photo_count']=table.PetID.map(counts).fillna(0).astype(int)
    table['default_photo_present']=table.PetID.isin(default.index)
    table['photo_count_mismatch']=table.actual_photo_count.ne(table.PhotoAmt)
    image_to_pet=images.set_index('image_key').PetID.to_dict()
    links=[];cross_pet=[]
    for row in edges.itertuples(index=False):
        ids=[image_to_pet[x] for x in [row.id_a,row.id_b] if x in image_to_pet]
        if len(ids)==2 and ids[0]!=ids[1]:
            links.append({'id_a':ids[0],'id_b':ids[1],'kind':row.kind})
        elif len(ids)==1 and row.cross_dataset:cross_pet.append(ids[0])
    # Exact normalized long descriptions can be shared templates, so these are
    # conservative grouping links, never asserted to identify the same animal.
    normalized=table.Description.fillna('').str.casefold().str.replace(r'\s+',' ',regex=True).str.strip()
    for _,group in table.loc[normalized.str.len().ge(40)].groupby(normalized[normalized.str.len().ge(40)]):
        ids=group.PetID.tolist()
        for x in ids[1:]:links.append({'id_a':ids[0],'id_b':x,'kind':'exact_normalized_text_candidate'})
    links=pd.DataFrame(links,columns=['id_a','id_b','kind']).drop_duplicates()
    links.to_csv(out/'listing_duplicate_links.csv',index=False)
    table['duplicate_component']=connected_groups(table,'PetID',edges=links)
    # Quarantine PF duplicate components touching prior breed datasets. This
    # preserves prior Oxford/Stanford holdouts if their encoder is reused.
    touched=set(table.loc[table.PetID.isin(cross_pet),'duplicate_component'])
    table['external_breed_overlap']=table.duplicate_component.isin(touched)
    # Rescue + photo + text components built on ALL records before exclusions.
    table['split_group']=connected_groups(table,'PetID','RescuerID',links)
    table['exclusion_reason']=''
    table.loc[table.Quantity.ne(1),'exclusion_reason']+='multi_pet_profile;'
    table.loc[table.schema_review_reason.ne(''),'exclusion_reason']+='schema_review;'
    table.loc[table.external_breed_overlap,'exclusion_reason']+='external_breed_photo_candidate;'
    eligible=table.source_partition.eq('train')&table.exclusion_reason.eq('')
    cohort=table.loc[eligible].copy().reset_index(drop=True)
    if cohort.split_group.nunique()<5:raise ValueError('Need five independent groups')
    cohort['split']=''
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=215)
    for k,(_,idx) in enumerate(splitter.split(cohort,cohort.AdoptionSpeed,cohort.split_group)):
        cohort.loc[idx,'split']={0:'test',1:'validation',2:'train',3:'train',4:'train'}[k]
    cohort.AdoptionSpeed=cohort.AdoptionSpeed.astype(int)
    verify_splits(cohort,'PetID','split_group');verify_splits(cohort,'PetID','RescuerID')
    if (cohort.groupby('duplicate_component').split.nunique()>1).any():raise ValueError('Duplicate leakage')
    for split,sub in cohort.groupby('split'):
        if set(sub.AdoptionSpeed)!=set(range(5)):raise ValueError(split+' lacks classes')
    cohort['slow_listing_risk_label']=cohort.AdoptionSpeed.isin([3,4]).astype(int)
    table['split']=table.PetID.map(cohort.set_index('PetID').split)
    table.loc[table.source_partition.eq('test'),'split']='official_unlabeled'
    table['split']=table.split.fillna('excluded')
    features=cohort[['PetID']+FEATURES].copy()
    features.to_csv(out/'features.csv',index=False)
    labels=cohort[['PetID','RescuerID','split_group','duplicate_component','split','AdoptionSpeed',
                   'slow_listing_risk_label','default_photo_present','outcome_text_review']].copy()
    labels.to_csv(out/'labels_and_splits.csv',index=False)
    # Descriptions and raw mutable fields live only in raw/ review tables.
    table.drop(columns=['Description','Name']).to_csv(out/'listing_review.csv',index=False)
    review=table.loc[table.source_partition.eq('train')].copy()
    review['single_primary_candidate']=(review.Quantity.eq(1)&review.Gender.isin([1,2])&
        review.Breed1.ne(0)&review.Breed2.eq(0)&review.schema_review_reason.eq('')&review.default_photo_present)
    review['named_category_candidate']=review.single_primary_candidate&~review.primary_breed_name.fillna('').str.contains(GENERIC_PATTERN)
    coverage=review.groupby(['Type','Breed1','primary_breed_name'],dropna=False).agg(
        all_train_listings=('PetID','size'),single_primary_candidates=('single_primary_candidate','sum'),
        named_category_candidates=('named_category_candidate','sum'))
    coverage.reset_index().to_csv(out/'breed_coverage.csv',index=False)
    processor=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUM),
        ('cat',OneHotEncoder(handle_unknown='ignore'),CAT)])
    processor.fit(features.loc[labels.split.eq('train'),FEATURES])
    array_audit={}
    for split in ['train','validation','test']:
        use=labels.split.eq(split)
        x=sparse.csr_matrix(processor.transform(features.loc[use,FEATURES]))
        if not np.isfinite(x.data).all():raise ValueError('Nonfinite feature')
        y=labels.loc[use].copy();y['matrix_row']=np.arange(len(y))
        sparse.save_npz(out/f'X_{split}.npz',x);y.to_csv(out/f'labels_{split}.csv',index=False)
        array_audit[split]={'shape':list(x.shape),'class_counts':y.AdoptionSpeed.value_counts().sort_index().to_dict(),
                            'slow_class_3_or_4_count':int(y.slow_listing_risk_label.sum()),
                            'with_default_photo':int(y.default_photo_present.sum())}
    joblib.dump(processor,out/'train_fitted_preprocessor.joblib')
    json_write(out/'feature_names.json',processor.get_feature_names_out().tolist())
    json_write(out/'feature_policy.json',{'used':FEATURES,'omitted':OMITTED,'fit_scope':'Internal train only',
        'media_policy':'Default photos exported as optional experimental input; no capture/upload history, not proven first-listing snapshots.',
        'deployment_gate':'Even retained listing fields lack independent historical versions; benchmark validity does not establish prospective deployment validity.'})
    image_export=[];export=Path(a.export_images);export.parent.mkdir(parents=True,exist_ok=True)
    if export.exists():raise ValueError('Image export exists; choose a new file')
    with zipfile.ZipFile(export,'w',zipfile.ZIP_STORED) as z:
        for row in cohort.loc[cohort.default_photo_present].itertuples(index=False):
            source=default.loc[row.PetID,'source_file']
            with Image.open(raw/source) as orig:
                im=ImageOps.exif_transpose(orig).convert('RGB');im.thumbnail((224,224),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(224,224),(128,128,128));canvas.paste(im,((224-im.width)//2,(224-im.height)//2))
                buf=io.BytesIO();canvas.save(buf,'JPEG',quality=90)
            member=f'{row.split}/{row.PetID}.jpg';z.writestr(member,buf.getvalue())
            image_export.append({'PetID':row.PetID,'split':row.split,'source_file':source,'prepared_member':member,
                                 'prepared_sha256':hashlib.sha256(buf.getvalue()).hexdigest()})
        z.writestr('README.txt','Restricted competition data. Use only as permitted by accepted Kaggle competition rules.\n'
                   'Optional default-photo benchmark subset; images not verified as first-listing-time snapshots.\n'
                   'Apply labels_and_splits.csv from the matching starter. Do not random-split photos.\n'
                   'RGB EXIF orientation, aspect-ratio resize without upscaling, grey letterbox 224x224, JPEG quality 90.\n')
        manifest=pd.DataFrame(image_export);z.writestr('image_manifest.csv',manifest.to_csv(index=False))
    manifest.to_csv(out/'prepared_image_manifest.csv',index=False)
    optional=audit_json_files(raw,table,images,out)
    counts=table.loc[table.source_partition.eq('train')].groupby('split_group').size()
    report={'actual_train_rows':len(train),'actual_official_test_rows':len(test),'official_test_has_target':False,
            'original_target_counts':train.AdoptionSpeed.value_counts().sort_index().to_dict(),
            'raw_missing':train.isna().sum().to_dict(),'quantity_one_raw':int(train.Quantity.eq(1).sum()),
            'train_schema_review_rows':int(review.schema_review_reason.ne('').sum()),
            'train_external_overlap_rows':int(review.external_breed_overlap.sum()),
            'eligible_labeled_rows':len(cohort),'internal_splits':array_audit,
            'group_count':int(cohort.split_group.nunique()),'largest_eligible_group':int(cohort.split_group.value_counts().max()),
            'image_counts':images.source_partition.value_counts().to_dict(),'decode_failures':int((~images.decode_ok).sum()),
            'train_photo_count_mismatch':int(review.photo_count_mismatch.sum()),
            'train_without_default_photo':int((~review.default_photo_present).sum()),
            'outcome_description_flag_train_rows':int(review.outcome_text_review.sum()),
            'single_primary_candidate_rows':int(review.single_primary_candidate.sum()),
            'named_category_candidate_rows':int(review.named_category_candidate.sum()),
            'named_candidate_categories_by_species':review.loc[review.named_category_candidate].groupby('Type').Breed1.nunique().to_dict(),
            'prepared_default_images':len(manifest),'prepared_zip_bytes':export.stat().st_size,
            'prepared_zip_sha256':hashlib.file_digest(export.open('rb'),'sha256').hexdigest(),
            'target':'AdoptionSpeed 0..4; optional slow_listing_risk_label=(class 3 or 4), NOT LOS days',
            'split_method':'5-fold StratifiedGroupKFold, seed 215; fold 0 test, 1 validation, 2..4 train; rescuer + image/text connected components before filtering',
            'grouping_scope':'All train and unlabeled test profiles included as identity bridges only; unlabeled data do not fit encoders or enter scores',
            'temporal_status':'No listing dates provided; NOT a temporal validation. Collection/feature-time provenance unverified.',
            'duplicate_status':'Machine exact/pHash candidates conservatively grouped; human identity review pending',
            'taxonomy_status':'Listing-owner labels, generic/mixed categories and imbalance; named candidates are NOT verified pure-breed images',
            'optional_json_files':sum(x['files'] for x in optional),'optional_json_malformed':sum(len(x['malformed']) for x in optional),
            'model_training':'Not run; preprocessing only; no model-performance claims'}
    json_write(out/'audit.json',report);print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
