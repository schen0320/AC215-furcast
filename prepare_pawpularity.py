"""Pawpularity image preparation. Requires authorized local original files.

As delivered on 2026-10-15 this script has synthetic guard-test coverage only;
the official download failed, so no real Pawpularity outputs are claimed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_STORED
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from audit_oxford_images import perceptual_hash
from find_petfinder_duplicates import find_pairs
from furcast_data import PAW, audit_pawpularity, connected_groups, require, verify_splits


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_labels(df):
    if set(df.columns) != set(['Id', 'Pawpularity'] + PAW):
        raise ValueError('Unexpected train.csv schema; review original version')
    audit_pawpularity(df)
    if df.empty or not df.Id.astype(str).str.fullmatch(r'[0-9a-f]{32}').all():
        raise ValueError('Empty table or unexpected profile Id format')
    if not (df.Pawpularity % 1 == 0).all():
        raise ValueError('Expected integer original Pawpularity scores')


def inspect_image(task):
    root, part, ident = task
    path = Path(root)/part/(ident+'.jpg')
    row = dict(key='pawpularity:'+part+'/'+ident, Id=ident,
               source_partition=part, source_file=f'{part}/{ident}.jpg')
    data = path.read_bytes()
    row.update(bytes=len(data), file_sha256=hashlib.sha256(data).hexdigest())
    try:
        with Image.open(io.BytesIO(data)) as original:
            im = ImageOps.exif_transpose(original).convert('RGB'); im.load()
            row.update(decode_ok=True, width=im.width, height=im.height,
                       pixel_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),
                       phash=f'{perceptual_hash(im):016x}')
    except Exception as exc:
        row.update(decode_ok=False, error=type(exc).__name__)
    return row


def group_and_split(images, external):
    """Exclude every Paw component touching prior data, preserving prior splits.

    Exact/near photo candidates are conservative links, not animal identity.
    Components are formed across ALL decoded photos before exclusions.
    """
    cols=['key','phash','pixel_sha256']
    require(external, cols)
    external=external[cols].copy()
    if external.empty or external.isna().any().any():
        raise ValueError('External fingerprints must be nonempty and complete')
    if external.key.str.startswith('pawpularity:').any():
        raise ValueError('External fingerprints must belong to previous sources')
    good=images.loc[images.decode_ok].copy()
    joined=pd.concat([external,good[cols]],ignore_index=True)
    if joined.key.duplicated().any():raise ValueError('Duplicate image key')
    edges=find_pairs(joined)
    joined['duplicate_component']=connected_groups(joined,'key',edges=edges)
    external_components=set(joined.loc[~joined.key.str.startswith('pawpularity:'),'duplicate_component'])
    result=images.merge(joined[['key','duplicate_component']],on='key',how='left',validate='one_to_one')
    result['split']='excluded';result['exclusion_reason']=''
    result.loc[~result.decode_ok,'exclusion_reason']='decode_failed'
    # Example test images never enter training, validation or real test.
    result.loc[result.source_partition.eq('test'),'exclusion_reason']='synthetic_example_only'
    cross=result.duplicate_component.isin(external_components)&result.source_partition.eq('train')
    result.loc[cross,'exclusion_reason']='cross_source_duplicate_candidate'
    keep=result.loc[result.exclusion_reason.eq('')].sort_values('key')
    if keep.duplicate_component.nunique()<5:
        raise ValueError('Fewer than five independent image components; no usable three-way split')
    trainval_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=215).split(keep,groups=keep.duplicate_component))
    trainval=keep.iloc[trainval_idx]
    train_idx,val_idx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=215).split(trainval,groups=trainval.duplicate_component))
    result.loc[keep.iloc[test_idx].index,'split']='test'
    result.loc[trainval.iloc[train_idx].index,'split']='train'
    result.loc[trainval.iloc[val_idx].index,'split']='validation'
    verify_splits(result,'key','duplicate_component')
    return result,edges


def prepare(raw, out, external_paths, export_images=None, size=300, workers=8):
    raw=Path(raw).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Choose a new output directory')
    if size<32 or workers<1:raise ValueError('Invalid image size or worker count')
    if not (raw/'train.csv').is_file():raise FileNotFoundError('Missing authorized train.csv')
    train=pd.read_csv(raw/'train.csv',dtype={'Id':str});validate_labels(train)
    expected=set(train.Id+'.jpg');actual={p.name for p in (raw/'train').glob('*.jpg')}
    if expected!=actual:
        raise ValueError(f'Train image/label mismatch: missing={len(expected-actual)}, orphan={len(actual-expected)}')
    # Test CSV is optional for a train-only delivery; when supplied it remains smoke-test only.
    test=pd.DataFrame(columns=['Id']+PAW)
    if (raw/'test.csv').exists():
        test=pd.read_csv(raw/'test.csv',dtype={'Id':str})
        if set(test.columns)!=set(['Id']+PAW):raise ValueError('Unexpected example test schema')
        if test.Id.isna().any() or test.Id.duplicated().any() or not test.Id.str.fullmatch(r'[0-9a-f]{32}').all():
            raise ValueError('Invalid example test IDs')
        if set(train.Id)&set(test.Id):raise ValueError('Example test/training Id overlap')
        if not test[PAW].isin([0,1]).all().all():raise ValueError('Nonbinary example metadata')
        if set(test.Id+'.jpg')!={p.name for p in (raw/'test').glob('*.jpg')}:
            raise ValueError('Example test images/IDs mismatch')
    elif list((raw/'test').glob('*.jpg')):
        raise ValueError('Example images present without test.csv')
    external=[];external_evidence=[]
    for p in external_paths:
        f=pd.read_csv(p,dtype={'phash':str,'pixel_sha256':str})
        if 'image_key' in f.columns:f=f.rename(columns={'image_key':'key'})
        if 'decode_ok' in f.columns:f=f.loc[f.decode_ok.eq(True)]
        require(f,['key','phash','pixel_sha256'])
        external.append(f[['key','phash','pixel_sha256']])
        external_evidence.append({'file':Path(p).name,'sha256':sha(p),'rows':len(f)})
    if not external:raise ValueError('Supply prior-source fingerprints for cross-source guard')
    external=pd.concat(external,ignore_index=True)
    tasks=[(str(raw),part,i) for part,df in [('train',train),('test',test)] for i in df.Id]
    if workers==1:rows=list(map(inspect_image,tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows=list(pool.map(inspect_image,tasks,chunksize=32))
    images=pd.DataFrame(rows)
    out.mkdir(parents=True)
    images.to_csv(out/'image_manifest.csv',index=False)
    try:
        splits,edges=group_and_split(images,external)
        edges.loc[edges.id_a.str.startswith('pawpularity:')|edges.id_b.str.startswith('pawpularity:')].to_csv(out/'duplicate_candidates.csv',index=False)
        splits.to_csv(out/'image_splits.csv',index=False)
        real=splits.loc[splits.source_partition.eq('train')].merge(train,on='Id',validate='one_to_one')
        real[['Id','Pawpularity','split','duplicate_component','exclusion_reason']].to_csv(out/'labels_and_splits.csv',index=False)
        # Hand labels are auxiliary targets, not assumed deployment inputs.
        real[['Id','split']+PAW].to_csv(out/'auxiliary_photo_labels.csv',index=False)
        bad_meta=real.Blur.eq(1)&real.Eyes.ne(0)
        real.loc[bad_meta,['Id','Blur','Eyes']].to_csv(out/'metadata_review.csv',index=False)
        conflicts=real.loc[real.decode_ok].groupby('pixel_sha256').Pawpularity.nunique()
        conflicting=set(conflicts[conflicts>1].index)
        real.loc[real.pixel_sha256.isin(conflicting),['Id','pixel_sha256','Pawpularity','duplicate_component']].to_csv(out/'identical_photo_different_scores.csv',index=False)
        active=real.loc[real.split.ne('excluded')].copy()
        active['prepared_member']=active['split']+'/'+active.Id+'.jpg'
        active[['Id','source_file','prepared_member','split','duplicate_component']].to_csv(out/'image_inputs.csv',index=False)
        for s in ['train','validation','test']:
            active.loc[active.split.eq(s),['Id','Pawpularity']].to_csv(out/f'labels_{s}.csv',index=False)
        report={
            'status':'completed_on_supplied_local_files','source_train_rows':len(train),
            'source_example_test_rows':len(test),'decode_failures':int((~images.decode_ok).sum()),
            'retained_rows':len(active),'split_counts':active.split.value_counts().to_dict(),
            'exclusion_counts':real.exclusion_reason.value_counts().to_dict(),
            'score_distribution_by_split':active.groupby('split').Pawpularity.describe().to_dict(),
            'metadata_positive_counts':train[PAW].sum().to_dict(),
            'blur_eyes_review_rows':int(bad_meta.sum()),
            'identical_pixel_groups_with_different_scores':len(conflicting),
            'train_csv_sha256':sha(raw/'train.csv'),'external_fingerprints':external_evidence,
            'split_rule':'Seed 215, GroupShuffleSplit approximately 60/20/20 groups; no target-based seed selection',
            'dedup_rule':'Exact oriented RGB pixels or 64-bit DCT pHash Hamming <=6, transitive components; exclude every component touching previous sources',
            'identity_limit':'No rescuer/animal/time IDs available. Different views of the same animal may escape. This is not prospective/temporal validation.',
            'score_conflicts':'Retained within the same image component; score can differ by profile exposure/context. Not silently averaged or relabeled.',
            'input_policy':'Photo pixels only. Id/path/split are bookkeeping; metadata are optional auxiliary targets. Never join AdoptionSpeed or LOS as this target.',
            'image_transform':f'EXIF oriented RGB, preserve aspect, no upscaling, centered gray letterbox {size}x{size}, JPEG quality90; no augmentation or model normalization',
            'model_training':'not performed','prepared_images_exported':False}
        if export_images:
            export=Path(export_images).resolve()
            if export.exists():raise FileExistsError('Image ZIP exists; preserve frozen output')
            export.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(suffix='.zip');os.close(fd)
            prepared=[]
            try:
                with ZipFile(tmp,'w',ZIP_STORED) as z:
                    for row in active.itertuples():
                        with Image.open(raw/row.source_file) as original:
                            im=ImageOps.exif_transpose(original).convert('RGB')
                            im.thumbnail((size,size),Image.Resampling.LANCZOS)
                            canvas=Image.new('RGB',(size,size),(127,127,127))
                            canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
                            b=io.BytesIO();canvas.save(b,format='JPEG',quality=90)
                        data=b.getvalue();z.writestr(row.prepared_member,data)
                        prepared.append({'Id':row.Id,'prepared_member':row.prepared_member,'sha256':hashlib.sha256(data).hexdigest()})
                with ZipFile(tmp) as z:
                    if z.testzip() is not None:raise ValueError('Export CRC failure')
                os.replace(tmp,export)
            finally:
                Path(tmp).unlink(missing_ok=True)
            pd.DataFrame(prepared).to_csv(out/'prepared_image_manifest.csv',index=False)
            report.update(prepared_images_exported=True,prepared_zip_sha256=sha(export))
        (out/'audit.json').write_text(json.dumps(report,indent=2,default=int)+'\n')
        return report
    except Exception as exc:
        (out/'run_failed.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,'message':str(exc)}))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True)
    p.add_argument('--external-manifest',action='append',required=True)
    p.add_argument('--export-images');p.add_argument('--size',type=int,default=300)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    print(json.dumps(prepare(a.raw,a.out,a.external_manifest,a.export_images,a.size,a.workers),indent=2,default=int))


if __name__=='__main__':main()
