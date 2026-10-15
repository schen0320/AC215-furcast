"""Decode official Oxford images and conservatively isolate duplicate candidates.

No image augmentation or label edits. Raw archives stay unchanged. Perceptual
matches are candidates, not confirmed same animals; the audit cannot establish
animal-level independence from photographs alone.
"""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.fft import dctn
from furcast_data import audit_oxford,connected_groups,write_json

def perceptual_hash(im):
    gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    low=dctn(gray,type=2,norm='ortho')[:8,:8].flatten()
    median=np.median(low[1:]);bits=low>median;bits[0]=False
    return sum(int(x)<<k for k,x in enumerate(bits))

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    annotations,annotation_audit=audit_oxford(args.annotations)
    rows=[]
    with tarfile.open(args.images,'r|gz') as t:
        for member in t:
            name=Path(member.name)
            if not member.isfile() or name.suffix.lower() not in {'.jpg','.jpeg','.png'} or name.name.startswith('._'):continue
            raw=t.extractfile(member).read()
            item={'image_id':name.stem,'archive_member':member.name,'file_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            try:
                with Image.open(io.BytesIO(raw)) as original:
                    im=ImageOps.exif_transpose(original).convert('RGB');im.load()
                    item.update(decode_ok=True,width=im.width,height=im.height,original_mode=original.mode,
                        pixel_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),
                        phash=f'{perceptual_hash(im):016x}')
            except Exception as e:item.update(decode_ok=False,error=type(e).__name__)
            rows.append(item)
            if len(rows)%1000==0:print('Images decoded',len(rows),flush=True)
    images=pd.DataFrame(rows)
    if images.image_id.duplicated().any():raise ValueError('Duplicate filename stems in archive')
    images.to_csv(out/'image_manifest.csv',index=False)
    joined=annotations.merge(images,on='image_id',how='left',validate='one_to_one',indicator=True)
    joined['image_ready']=joined['_merge'].eq('both') & joined.decode_ok.fillna(False).astype(bool)
    valid=joined.loc[joined.image_ready].reset_index(drop=True)
    edges=[]
    # At most ~27 million comparisons for 7,349 images. 64-bit XOR is deterministic.
    ids=valid.image_id.tolist();phashes=[int(x,16) for x in valid.phash];pixels=valid.pixel_sha256.tolist()
    for a in range(len(ids)):
        for b in range(a):
            distance=(phashes[a]^phashes[b]).bit_count()
            exact=pixels[a]==pixels[b]
            if exact or distance<=6:edges.append({'id_a':ids[a],'id_b':ids[b],
                'kind':'exact_pixels' if exact else 'perceptual_candidate','phash_distance':distance})
    edge_frame=pd.DataFrame(edges,columns=['id_a','id_b','kind','phash_distance'])
    edge_frame.to_csv(out/'duplicate_candidates.csv',index=False)
    joined['duplicate_component']=connected_groups(joined,'image_id',edges=edge_frame)
    joined['clean_split']=joined['split'];joined['exclusion_reason']=''
    joined.loc[~joined.image_ready,['clean_split','exclusion_reason']]=['excluded','missing_or_undecodable']
    # Keep official test membership. Remove development-side candidates that overlap
    # test, and remove train-side candidates that overlap validation. Never move a
    # held-out image into training. Same-split duplicates remain with a component ID.
    for _,g in joined.groupby('duplicate_component'):
        splits=set(g.loc[g.image_ready,'split'])
        drop=g.index[g['split'].ne('test')] if 'test' in splits else g.index[g['split'].eq('train')] if 'validation' in splits else []
        if len(splits)>1:
            joined.loc[drop,['clean_split','exclusion_reason']]=['excluded','cross_split_duplicate_candidate']
    # Exact pixels with conflicting labels cannot be a trustworthy training target.
    for _,g in joined.loc[joined.image_ready].groupby('pixel_sha256'):
        if g.class_id.nunique()>1:
            joined.loc[g.index,['clean_split','exclusion_reason']]=['excluded','exact_pixel_label_conflict']
    kept=joined.loc[joined.clean_split.ne('excluded')]
    if (kept.groupby('duplicate_component').clean_split.nunique()>1).any():raise ValueError('Duplicate component crosses clean split')
    joined.drop(columns=['_merge']).to_csv(out/'image_splits.csv',index=False)
    labels=joined.set_index('image_id').class_id.to_dict()
    split=joined.set_index('image_id')['split'].to_dict()
    annotation_audit['image_bytes_status']='Official image archive downloaded and decoded; see this image audit'
    report={'archive_image_files':len(images),'listed_images':len(joined),'listed_image_missing':int(joined['_merge'].eq('left_only').sum()),
      'archive_decode_failures':int((~images.decode_ok).sum()),'listed_decode_or_missing_failures':int((~joined.image_ready).sum()),
      'extra_unlisted_image_files':int((~images.image_id.isin(joined.image_id)).sum()),
      'exact_duplicate_pairs':sum(e['kind']=='exact_pixels' for e in edges),'perceptual_candidate_pairs':sum(e['kind']=='perceptual_candidate' for e in edges),
      'cross_split_candidate_pairs':sum(split[e['id_a']]!=split[e['id_b']] for e in edges),
      'candidate_pairs_with_different_class':sum(labels[e['id_a']]!=labels[e['id_b']] for e in edges),
      'original_split_counts':joined.split.value_counts().to_dict(),'clean_split_counts':joined.clean_split.value_counts().to_dict(),
      'clean_split_class_counts':kept.groupby('clean_split').class_id.nunique().to_dict(),
      'candidate_rule':'64-bit DCT pHash Hamming <=6 OR exact oriented RGB pixels; candidates are not confirmed identities',
      'scope':'No global guarantee of animal independence. Crops/other photographs of the same animal and cross-dataset duplicates may be missed.',
      'manual_review':'Pending for perceptual candidates; development-side cross-split candidates conservatively excluded without changing official test membership.',
      'annotation_audit':annotation_audit}
    write_json(out/'image_audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
