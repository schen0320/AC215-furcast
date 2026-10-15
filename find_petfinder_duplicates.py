"""Exact radius-6 pHash search using seven disjoint bit bands, not ANN.

At <=6 changed bits, at least one of seven bands is unchanged. Candidate
retrieval is exhaustive for this Hamming radius; visual identity is not.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd


def find_pairs(frame, radius=6):
    if radius!=6: raise ValueError('Seven-band implementation requires radius 6')
    values=[int(x,16) for x in frame.phash]
    packed=np.asarray(values,dtype=np.uint64)
    keys=frame.key.tolist(); pixels=frame.pixel_sha256.tolist()
    bands=[defaultdict(list) for _ in range(7)]
    exact=defaultdict(list); rows=[]
    for i,value in enumerate(values):
        candidates=set()
        for k,index in enumerate(bands):
            # Last band includes bits 54..63 (10 bits); other bands have 9.
            mask=1023 if k==6 else 511
            candidates.update(index[(value>>(9*k))&mask])
        candidates.update(exact[pixels[i]])
        if candidates:
            idx=np.fromiter(candidates,dtype=np.int64)
            distance=np.bitwise_count(packed[idx]^np.uint64(value))
            for j,d in zip(idx,distance):
                if d<=radius or pixels[i]==pixels[j]:
                    rows.append({'id_a':keys[i],'id_b':keys[j],'phash_distance':int(d),
                                 'kind':'exact_pixels' if pixels[i]==pixels[j] else 'perceptual_candidate',
                                 'cross_dataset':keys[i].split(':',1)[0]!=keys[j].split(':',1)[0]})
        for k,index in enumerate(bands):
            mask=1023 if k==6 else 511;index[(value>>(9*k))&mask].append(i)
        exact[pixels[i]].append(i)
        if (i+1)%20000==0:print('Compared',i+1,'of',len(frame),'candidate pairs',len(rows),flush=True)
    return pd.DataFrame(rows,columns=['id_a','id_b','phash_distance','kind','cross_dataset'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--petfinder-manifest',required=True)
    p.add_argument('--breed-manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    pf=pd.read_csv(a.petfinder_manifest,dtype={'phash':str})
    pf=pf.loc[pf.decode_ok].rename(columns={'image_key':'key'})
    breed=pd.read_csv(a.breed_manifest,dtype={'phash':str})
    joined=pd.concat([breed[['key','phash','pixel_sha256']],pf[['key','phash','pixel_sha256']]],ignore_index=True)
    if joined.key.duplicated().any():raise ValueError('Duplicate image key')
    edges=find_pairs(joined);edges.to_csv(out/'all_image_candidates.csv',index=False)
    mask=edges.id_a.str.startswith('petfinder:')|edges.id_b.str.startswith('petfinder:')
    involved=edges.loc[mask]; involved.to_csv(out/'petfinder_image_candidates.csv',index=False)
    report={'images_compared':len(joined),'petfinder_images':len(pf),
            'all_candidate_pairs':len(edges),'petfinder_involved_pairs':len(involved),
            'cross_dataset_petfinder_pairs':int(involved.cross_dataset.sum()),
            'petfinder_exact_pairs':int(involved.kind.eq('exact_pixels').sum()),
            'manual_review':'Not performed. Perceptual pairs are conservatively grouped, not called confirmed duplicate animals.',
            'rule':'Exact oriented RGB pixels OR 64-bit DCT pHash Hamming <=6; seven-band exhaustive radius search',
            'limits':'Does not identify all crops or different photos of one animal; does not prove animal independence'}
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
