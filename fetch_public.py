"""Acquire fresh Austin snapshots into a NEW directory; never overwrite frozen data.

Usage: python fetch_public.py --out snapshots/2026-10-15
No Kaggle downloads; users obtain those files after reviewing competition rules.
"""
import argparse,datetime,hashlib,json,urllib.parse,urllib.request
from pathlib import Path

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FurCast-AC215/0.1'}),timeout=55) as r:
        return r.read()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.out)
    if root.exists():raise SystemExit('Use a new output directory to preserve existing snapshots.')
    root.mkdir(parents=True)
    manifest=[]
    try:
        for rid in ['wter-evkm','9t4d-g238','pyqf-r2dc','gsvs-ypi7']:
            meta_url=f'https://data.austintexas.gov/api/views/{rid}.json'
            before=get(meta_url);metadata=json.loads(before)
            base=f'https://data.austintexas.gov/resource/{rid}.json'
            count_url=base+'?'+urllib.parse.urlencode({'$select':'count(*)'})
            count=int(json.loads(get(count_url))[0]['count'])
            url=base+'?'+urllib.parse.urlencode({'$limit':count+1,'$order':':id'})
            raw=get(url);rows=json.loads(raw)
            after=json.loads(get(meta_url));count_after=int(json.loads(get(count_url))[0]['count'])
            if len(rows)!=count or count!=count_after or metadata.get('rowsUpdatedAt')!=after.get('rowsUpdatedAt'):
                raise RuntimeError(f'{rid}: snapshot changed or truncated. Retain prior files; retry in a new directory.')
            (root/f'{rid}.json').write_bytes(raw);(root/f'{rid}_metadata.json').write_bytes(before)
            manifest.append({'dataset_id':rid,'url':url,'count':count,'sha256':hashlib.sha256(raw).hexdigest(),
                'bytes':len(raw),'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'rowsUpdatedAt':metadata.get('rowsUpdatedAt'),'licenseId':metadata.get('licenseId')})
            print(rid,count,'rows')
        (root/'manifest.json').write_text(json.dumps({'complete':True,'tables':manifest},indent=2))
    except Exception as e:
        (root/'manifest.json').write_text(json.dumps({'complete':False,'tables':manifest,'error':str(e)},indent=2))
        raise

if __name__=='__main__':main()
