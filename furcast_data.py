"""FurCast milestone data tools. Target tables and feature tables stay separate.

Python >=3.11; pandas/numpy/scikit-learn/Pillow. No credentials or automatic Kaggle access.
"""
from __future__ import annotations
import argparse, hashlib, json, re, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

TZ = 'America/Chicago'
PF_CAT = ['Type','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
          'FurLength','Vaccinated','Dewormed','Sterilized','Health','State']
PF_NUM = ['Age','Fee']
PF_REQUIRED = PF_CAT + PF_NUM + ['PetID','RescuerID','Quantity','PhotoAmt','VideoAmt','Description','Name','AdoptionSpeed']
PAW = ['Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur']
AAC_FEATURES = ['species','intake_type','intake_condition','sex_at_intake','breed_at_intake',
                'color_at_intake','age_days_approx','intake_month']
NEW_ADOPTIONS = {'Adopted','Adopted Altered','Adopted Offsite(Altered)','Adopted Offsite',
                'Adopted Unaltered','Adopted Offsite(Unaltered)','Adopted to Finder'}
NEW_OTHER_EXITS = {'Transfer Out','Reclaimed','Euthanized','DOA','Return to Habitat',
 'Unassisted Death','Redemption (Offsite)','Unassisted Death - In Transit',
 'Unassisted Death - In Foster','Lost','DOA - Final Disposition','DOA - Released to Owner',
 'Interred','Euthanasia - RTO','Returned To Owner','Cremated','Escaped','Stolen','Cremation - Communal'}
OLD_EXITS = {'Adoption','Transfer','Return to Owner','Euthanasia','Died','Rto-Adopt','Disposal','Missing','Relocate','Stolen','Lost'}

def require(df, cols):
    missing = set(cols)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')

def profile(df):
    return {'rows':len(df),'columns':list(df.columns),'duplicate_rows':int(df.duplicated().sum()),
            'null_counts':{k:int(v) for k,v in df.isna().sum().items()},
            'repeated_animal_rows':int(df.animal_id.duplicated().sum()) if 'animal_id' in df else None}

def local_times(s):
    # Archive outcomes mix naive times and offset-bearing strings. Do not silently
    # combine clock conventions: offset-bearing and DST-ambiguous rows are quarantined.
    text=s.astype('string')
    offsets=text.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',na=False)
    parsed=pd.to_datetime(text.mask(offsets),format='mixed',errors='coerce')
    result=parsed.dt.tz_localize(TZ,ambiguous='NaT',nonexistent='NaT')
    return result, offsets | result.isna()

def age_days(s):
    parts=s.astype('string').str.lower().str.extract(r'^\s*(\d+)\s+(day|week|month|year)s?\s*$')
    return pd.to_numeric(parts[0],errors='coerce')*parts[1].map({'day':1.,'week':7.,'month':365.25/12,'year':365.25})

def build_austin(intakes, outcomes, era, cutoff, horizon=30):
    """Conservative episodes; unmatched is NOT automatically a confirmed censor.

    Intake records define the cohort. Current era joins animal_id + intake_date;
    archive uses one outcome in [intake, next intake). Ambiguities are retained
    with null labels. A known unreliable event quarantines that animal's timeline.
    """
    if era not in {'legacy','current'}: raise ValueError('era must be legacy/current')
    if horizon!=30: raise ValueError('This starter implements the explicitly named 30-day target only')
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None: raise ValueError('cutoff must have timezone offset')
    cutoff=cutoff.tz_convert(TZ)
    i=intakes.drop_duplicates().copy(); o=outcomes.drop_duplicates().copy()
    i_date='datetime' if era=='legacy' else 'source_date'
    o_date='datetime' if era=='legacy' else 'outcome_date'
    cause='outcome_type' if era=='legacy' else 'outcome_status'
    require(i,['animal_id',i_date]);require(o,['animal_id',o_date,cause])
    for frame in [i,o]:
        if frame.animal_id.isna().any(): raise ValueError('Missing animal_id; quarantine before processing')
        frame['animal_id']=frame.animal_id.astype(str)
    i['prediction_time'],bad_i=local_times(i[i_date]);o['outcome_time'],bad_o=local_times(o[o_date])
    bad_animals=set(i.loc[bad_i,'animal_id'])|set(o.loc[bad_o,'animal_id'])
    i=i.loc[i.prediction_time.notna() & (i.prediction_time<=cutoff)].copy()
    # Snapshot boundary: future outcomes do not enter labels or pairing.
    o=o.loc[o.outcome_time.notna() & (o.outcome_time<=cutoff)].copy()
    duplicate_i=i.duplicated(['animal_id','prediction_time'],keep=False)
    bad_animals |= set(i.loc[duplicate_i,'animal_id'])
    i=i.sort_values(['animal_id','prediction_time']).drop_duplicates(['animal_id','prediction_time']).reset_index(drop=True)
    i['episode_id']=[f'austin_{era}:{a}:{t.isoformat()}' for a,t in zip(i.animal_id,i.prediction_time)]
    i['next_intake']=i.groupby('animal_id').prediction_time.shift(-1)
    i['row_id']=np.arange(len(i))
    i['status']='unmatched_unverified'
    i.loc[i.next_intake.notna(),'status']='unresolved_before_reintake'
    i.loc[i.animal_id.isin(bad_animals),'status']='quarantine_timeline'
    i['outcome_time']=pd.Series(pd.NaT,index=i.index,dtype=f'datetime64[ns, {TZ}]')
    i['outcome_type']=pd.NA
    good=i.loc[~i.animal_id.isin(bad_animals)]
    usable_o=o.loc[~o.animal_id.isin(bad_animals)].copy()
    if era=='legacy':
        if len(good) and len(usable_o):
            matched=pd.merge_asof(usable_o.sort_values('outcome_time'),
                good[['animal_id','prediction_time','row_id']].sort_values('prediction_time'),
                by='animal_id',left_on='outcome_time',right_on='prediction_time',direction='backward')
        else: matched=usable_o.assign(row_id=np.nan)
        # Outcomes at a re-intake instant have ambiguous ordering: quarantine that match.
        boundary=good.loc[good.groupby('animal_id').cumcount().gt(0),['animal_id','prediction_time']]
        boundary_keys=set(zip(boundary.animal_id,boundary.prediction_time))
        matched['boundary_tie']=[(a,t) in boundary_keys for a,t in zip(matched.animal_id,matched.outcome_time)]
    else:
        require(usable_o,['intake_date','days_in_shelter'])
        usable_o['prediction_time'],bad_link=local_times(usable_o.intake_date)
        matched=usable_o.merge(good[['animal_id','prediction_time','row_id']],how='left',on=['animal_id','prediction_time'],validate='many_to_one')
        matched['boundary_tie']=False
    counts=matched.row_id.value_counts()
    multiple=set(counts[counts>1].index.astype(int))
    i.loc[i.row_id.isin(multiple),'status']='quarantine_multiple_outcomes'
    unique=matched.loc[matched.row_id.notna() & ~matched.row_id.isin(multiple)]
    valid_exits=OLD_EXITS if era=='legacy' else NEW_ADOPTIONS|NEW_OTHER_EXITS
    for row in unique.to_dict('records'):
        k=int(row['row_id']);event=row[cause];end=row['outcome_time'];start=i.at[k,'prediction_time'];next_i=i.at[k,'next_intake']
        if row['boundary_tie'] or end<start or (pd.notna(next_i) and end>=next_i):
            i.at[k,'status']='quarantine_time_order';continue
        if pd.isna(event) or event not in valid_exits:
            i.at[k,'status']='quarantine_unknown_outcome';continue
        i.at[k,'outcome_time']=end;i.at[k,'outcome_type']=event;i.at[k,'status']='paired'
    i['duration_days']=(i.outcome_time-i.prediction_time).dt.total_seconds()/86400
    i['followup_candidate_days']=(cutoff-i.prediction_time).dt.total_seconds()/86400
    i['event_observed']=i.status.eq('paired')
    i['event_cause']=np.where(i.outcome_type.isin({'Adoption'} if era=='legacy' else NEW_ADOPTIONS),'adoption',
                            np.where(i.event_observed,'other_exit','unresolved'))
    # No numerical days are invented for adoption-speed classes or missing outcomes.
    i['long_stay_30d']=pd.Series(pd.NA,index=i.index,dtype='Int64')
    paired=i.event_observed
    i.loc[paired,'long_stay_30d']=(i.loc[paired,'duration_days']>horizon).astype(int)
    i['label_available_at']=i.outcome_time.copy()  # conservative: observed event, not assumed status log
    i['snapshot_cutoff']=cutoff;i['source_dataset']='austin_'+era
    if era=='legacy':
        aliases={'animal_type':'species','intake_type':'intake_type','intake_condition':'intake_condition',
                 'sex_upon_intake':'sex_at_intake','breed':'breed_at_intake','color':'color_at_intake'}
        age=age_days(i.age_upon_intake)
    else:
        aliases={'type':'species','source_name':'intake_type','intake_health_condition':'intake_condition',
                 'sex':'sex_at_intake','primary_breed':'breed_at_intake','primary_color':'color_at_intake'}
        dob,bad_dob=local_times(i.date_of_birth)
        age=(i.prediction_time-dob).dt.total_seconds()/86400
        age=age.mask(bad_dob | (age<0))
    features=pd.DataFrame({'episode_id':i.episode_id})
    for raw,name in aliases.items():features[name]=i[raw].fillna('__MISSING__').astype(str)
    features['species']=features.species.replace({'Kitten':'Cat','Puppy':'Dog'})
    features['age_days_approx']=age;features['intake_month']=i.prediction_time.dt.month
    i['species']=features.species
    i['cohort_eligible']=i.species.isin(['Cat','Dog'])
    if era=='legacy': i['cohort_eligible'] &= ~i.intake_type.isin(['Euthanasia Request']) & ~i.intake_condition.eq('Dead')
    else: i['cohort_eligible'] &= ~features.intake_type.str.contains('Euthanasia Request',case=False) & ~features.intake_condition.str.contains(r'\b(?:dead|DOA)\b',case=False)
    columns=['episode_id','source_dataset','animal_id','species','prediction_time','outcome_time','snapshot_cutoff',
             'duration_days','followup_candidate_days','event_observed','event_cause','outcome_type','status',
             'long_stay_30d','label_available_at','cohort_eligible']
    report={'raw_intakes':profile(intakes),'raw_outcomes':profile(outcomes),
            'cutoff':str(cutoff),'horizon_days':horizon,'bad_timeline_animal_count':len(bad_animals),
            'status_counts':i.status.value_counts().to_dict(),'unmatched_outcome_records':int(matched.row_id.isna().sum()),
            'baseline_cat_dog_paired_rows':int((i.cohort_eligible & paired).sum()),
            'censoring_policy':'Unmatched records are unresolved, not confirmed still in care; no survival cohort claimed.',
            'timestamp_policy':'Naive Austin local times; offset-bearing or DST-ambiguous events quarantine the animal timeline.'}
    return i[columns],features,report

def temporal_split(labels, validation_start, test_start):
    """Strict event-time and animal separation. No random mixing of repeat admissions."""
    v=pd.Timestamp(validation_start);t=pd.Timestamp(test_start)
    if v.tzinfo is None or t.tzinfo is None or v>=t:raise ValueError('ordered timezone-aware boundaries required')
    d=labels.copy();d['split']='excluded'
    # Boundaries based on all intake episodes, including unlabeled ones, not only observed exits.
    periods=pd.Series(np.where(d.prediction_time<v,'train',np.where(d.prediction_time<t,'validation','test')),index=d.index)
    crossing=periods.groupby(d.animal_id).transform('nunique')>1
    eligible=d.cohort_eligible & d.event_observed & d.long_stay_30d.notna() & ~crossing
    for name,end in [('train',v),('validation',t),('test',None)]:
        mask=eligible & periods.eq(name)
        if end is not None:mask &= d.label_available_at<end
        d.loc[mask,'split']=name
    d['split_exclusion_reason']=np.where(crossing,'animal_crosses_time_boundary',
        np.where(~eligible,'ineligible_or_unresolved',np.where(d.split.eq('excluded'),'label_not_available_before_boundary','')))
    verify_splits(d,'episode_id','animal_id')
    assert (d.loc[d.split.eq('train'),'label_available_at']<v).all()
    assert (d.loc[d.split.eq('validation'),'label_available_at']<t).all()
    return d

def verify_splits(df,key,group):
    require(df,[key,group,'split'])
    if df[key].isna().any() or df[key].duplicated().any(): raise ValueError('Missing or duplicate record ID')
    x=df.loc[df.split.isin(['train','validation','test'])]
    if (x.groupby(group)['split'].nunique()>1).any():raise ValueError('Cross-split group leakage')
    if set(x.split.unique()) != {'train','validation','test'}:raise ValueError('Need nonempty train/validation/test')

def connected_groups(df,id_col,group_col=None,edges=None):
    """Union rescuer + reviewed duplicate-image/text links transitively before splitting."""
    ids=df[id_col].astype(str).tolist();parent={x:x for x in ids}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        if a not in parent or b not in parent:raise ValueError('Duplicate edge refers to unknown ID')
        x,y=root(a),root(b);parent[max(x,y)]=min(x,y)
    if group_col:
        if df[group_col].isna().any() or df[group_col].astype(str).str.strip().eq('').any():
            raise ValueError('Missing group ID: resolve/quarantine before splitting')
        for _,sub in df.groupby(group_col):
            group=sub[id_col].astype(str).tolist()
            for x in group[1:]:union(group[0],x)
    if edges is not None:
        require(edges,['id_a','id_b'])
        for a,b in edges[['id_a','id_b']].astype(str).itertuples(index=False,name=None):union(a,b)
    return pd.Series([root(x) for x in ids],index=df.index)

def prepare_petfinder(df,edges=None):
    require(df,PF_REQUIRED)
    if df.PetID.isna().any() or df.PetID.duplicated().any():raise ValueError('PetID must be unique/nonmissing')
    for c in PF_CAT+PF_NUM+['Quantity','PhotoAmt','VideoAmt','AdoptionSpeed']:
        df=df.copy() if c==PF_CAT[0] else df
        df[c]=pd.to_numeric(df[c],errors='raise')
        if df[c].isna().any() or (df[c]<0).any() or ((df[c]%1)!=0).any():raise ValueError(f'Invalid integer field {c}')
    ranges={'Type':{1,2},'Gender':{1,2,3},'MaturitySize':{0,1,2,3,4},'FurLength':{0,1,2,3},
            'Vaccinated':{1,2,3},'Dewormed':{1,2,3},'Sterilized':{1,2,3},'Health':{0,1,2,3},'AdoptionSpeed':set(range(5))}
    for c,allowed in ranges.items():
        if not df[c].isin(allowed).all():raise ValueError(f'Invalid enum {c}')
    if not df.Quantity.ge(1).all():raise ValueError('Quantity must be positive')
    df['split_group']=connected_groups(df,'PetID','RescuerID',edges)
    original=len(df);df=df.loc[df.Quantity.eq(1)].copy().reset_index(drop=True)
    if df.split_group.nunique()<5:raise ValueError('Need >=5 independent groups for 60/20/20 split')
    df['fold']=-1
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=215)
    for k,(_,idx) in enumerate(splitter.split(df,df.AdoptionSpeed,df.split_group)):df.loc[idx,'fold']=k
    df['split']=df.fold.map({0:'test',1:'validation',2:'train',3:'train',4:'train'})
    verify_splits(df,'PetID','split_group')
    # Flag whole descriptions rather than claiming regex removal makes snapshots leakage-free.
    pattern=r'(?i)\b(?:adopted|rehomed|re-homed|found (?:a |their )?(?:new |forever )?home|no longer available)\b|已[被获]?领养|已送养|已找到新家'
    df['outcome_text_review']=df.Description.fillna('').str.contains(pattern,regex=True)
    df['slow_listing_risk_label']=df.AdoptionSpeed.isin([3,4]).astype(int)
    for split,sub in df.groupby('split'):
        if set(sub.AdoptionSpeed)!=set(range(5)):raise ValueError(f'{split} lacks classes: adjust group split before training')
    labels=df[['PetID','RescuerID','split_group','split','AdoptionSpeed','slow_listing_risk_label','outcome_text_review']]
    # High-change media counts/text/health snapshots still need listing-time provenance audit.
    X=df[['PetID']+PF_CAT+PF_NUM].copy()
    report={'input_rows':original,'quantity_one_rows':len(df),'outcome_text_flags':int(df.outcome_text_review.sum()),
            'split_class_counts':df.groupby(['split','AdoptionSpeed']).size().to_dict(),
            'duplicate_audit':'supplied edges grouped; caller must document machine/human review provenance' if edges is not None else 'NOT COMPLETE; split is provisional',
            'feature_provenance':'Listing-time snapshots unverified; benchmark only, not proven prospective performance'}
    report['split_class_counts']={str(k):int(v) for k,v in report['split_class_counts'].items()}
    return labels,X,report

def audit_pawpularity(df):
    require(df,['Id','Pawpularity']+PAW)
    if df.Id.isna().any() or df.Id.duplicated().any():raise ValueError('Invalid Id')
    if not pd.to_numeric(df.Pawpularity,errors='raise').between(1,100).all():raise ValueError('Unexpected Pawpularity range; check official version')
    for c in PAW:
        if not df[c].isin([0,1]).all():raise ValueError(f'Nonbinary/missing {c}')
    return profile(df)

def audit_oxford(archive):
    with tarfile.open(archive) as t:
        names=set(t.getnames())
        def rows(name):
            lines=t.extractfile('annotations/'+name+'.txt').read().decode().splitlines()
            return pd.DataFrame([x.split() for x in lines if x and not x.startswith('#')],columns=['image_id','class_id','species_id','breed_id'])
        a,b,c=rows('list'),rows('trainval'),rows('test')
        if set(b.image_id)&set(c.image_id):raise ValueError('Official split overlap')
        if set(a.image_id)!=set(b.image_id)|set(c.image_id):raise ValueError('Split/list mismatch')
        if a.image_id.duplicated().any():raise ValueError('Duplicate image IDs')
        a['split']=np.where(a.image_id.isin(c.image_id),'test','trainval')
        tr,va=train_test_split(b.image_id,test_size=0.2,stratify=b.class_id,random_state=215)
        a.loc[a.image_id.isin(tr),'split']='train'
        a.loc[a.image_id.isin(va),'split']='validation'
        a['trimap_present']=a.image_id.map(lambda x:f'annotations/trimaps/{x}.png' in names)
        a['head_xml_present']=a.image_id.map(lambda x:f'annotations/xmls/{x}.xml' in names)
        return a,{'list_rows':len(a),'trainval_rows':len(b),'test_rows':len(c),'classes':a.class_id.nunique(),
          'breed_count_by_species':a.groupby('species_id').breed_id.nunique().to_dict(),
          'missing_trimaps':int((~a.trimap_present).sum()),'missing_head_xml':int((~a.head_xml_present).sum()),
          'provisional_split_counts':a.split.value_counts().to_dict(),
          'image_bytes_status':'This annotation-only command did not inspect image bytes; consult the separate image audit before using splits',
          'readme_issues':'Breed counts reversed in README; archive research-only note differs from live CC BY-SA 4.0 page.'}

def feature_guard(columns,task):
    allowed={'austin':AAC_FEATURES,'petfinder':PF_CAT+PF_NUM}[task]
    if set(columns)-set(allowed):raise ValueError('Feature allowlist violation: '+str(sorted(set(columns)-set(allowed))))

def image_manifest(root):
    """Exact pixel duplicate candidates; perceptual near-duplicates still need review."""
    from PIL import Image,ImageOps
    rows=[]
    for path in sorted(Path(root).rglob('*')):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
        item={'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.load()
                item.update(width=im.width,height=im.height,pixel_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),decode_ok=True)
        except Exception as e:item.update(decode_ok=False,error=type(e).__name__)
        rows.append(item)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('austin');a.add_argument('--intakes',required=True);a.add_argument('--outcomes',required=True)
    a.add_argument('--era',choices=['legacy','current'],required=True);a.add_argument('--cutoff',required=True)
    a.add_argument('--validation-start');a.add_argument('--test-start');a.add_argument('--out',required=True)
    f=sub.add_parser('petfinder');f.add_argument('--csv',required=True);f.add_argument('--edges');f.add_argument('--out',required=True)
    o=sub.add_parser('oxford');o.add_argument('--annotations',required=True);o.add_argument('--out',required=True)
    im=sub.add_parser('images');im.add_argument('--root',required=True);im.add_argument('--out',required=True)
    pw=sub.add_parser('pawpularity');pw.add_argument('--csv',required=True);pw.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.command=='austin':
        labels,X,report=build_austin(pd.read_json(args.intakes,convert_dates=False),pd.read_json(args.outcomes,convert_dates=False),args.era,args.cutoff)
        if bool(args.validation_start)!=bool(args.test_start):raise ValueError('Pass both split dates')
        if args.validation_start:
            labels=temporal_split(labels,args.validation_start,args.test_start)
            report['split_counts']=labels.split.value_counts().to_dict()
            report['split_positive_counts']=labels.groupby('split').long_stay_30d.sum().to_dict()
        feature_guard(X.columns.drop('episode_id'),'austin')
        labels.to_csv(out/'labels_and_splits.csv',index=False);X.to_csv(out/'intake_features.csv',index=False)
        write_json(out/'audit.json',report)
    elif args.command=='petfinder':
        labels,X,report=prepare_petfinder(pd.read_csv(args.csv),pd.read_csv(args.edges) if args.edges else None)
        feature_guard(X.columns.drop('PetID'),'petfinder')
        labels.to_csv(out/'labels_and_splits.csv',index=False);X.to_csv(out/'benchmark_features.csv',index=False);write_json(out/'audit.json',report)
    elif args.command=='oxford':
        m,r=audit_oxford(args.annotations);m.to_csv(out/'annotation_manifest.csv',index=False);write_json(out/'audit.json',r)
    elif args.command=='pawpularity':write_json(out/'audit.json',audit_pawpularity(pd.read_csv(args.csv)))
    else:image_manifest(args.root).to_csv(out/'image_manifest.csv',index=False)

if __name__=='__main__':main()
