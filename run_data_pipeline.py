"""One entry point; missing inputs remain explicit instead of fake success.

python run_data_pipeline.py --out results_rebuild --oxford-images /path/to/oxford_images.tar.gz
"""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,help='New output directory; existing paths are not overwritten')
    p.add_argument('--data-root',type=Path,help='Austin JSONs and Oxford annotations; defaults to ./raw beside this script')
    p.add_argument('--reference-results',type=Path,help='Prior frozen image manifests; defaults to ./results beside this script')
    p.add_argument('--skip-austin',action='store_true',help='Run only the requested image/competition stages')
    p.add_argument('--oxford-images');p.add_argument('--petfinder-csv');p.add_argument('--pawpularity-csv');p.add_argument('--reviewed-edges')
    p.add_argument('--stanford-raw',help='Directory containing the authorized official Stanford archives')
    p.add_argument('--petfinder-raw',help='Full authorized official PetFinder extraction; runs image audit, dedup and conservative preparation')
    p.add_argument('--pawpularity-raw',help='Authorized Pawpularity extraction; requires train.csv and train images, rejects incomplete data')
    a=p.parse_args();root=Path(__file__).resolve().parent;out=Path(a.out).resolve()
    data=(a.data_root or root/'raw').resolve()
    reference=(a.reference_results or root/'results').resolve()
    if a.pawpularity_raw and a.pawpularity_csv:raise SystemExit('Choose full --pawpularity-raw OR schema-only --pawpularity-csv')
    if a.petfinder_raw and a.petfinder_csv:raise SystemExit('Choose full --petfinder-raw OR provisional --petfinder-csv')
    # Validate requested dependencies before starting any expensive stage.
    required=[]
    for path in [a.petfinder_csv,a.pawpularity_csv,a.reviewed_edges]:
        if path:required.append(Path(path))
    if a.oxford_images:required.extend([Path(a.oxford_images),data/'oxford_annotations.tar.gz'])
    if a.stanford_raw:
        required.extend(Path(a.stanford_raw)/x for x in ['images.tar','annotation.tar','lists.tar','download_manifest.json'])
        if not a.oxford_images:required.append(reference/'oxford/image_splits.csv')
    if a.petfinder_raw:
        required.extend(Path(a.petfinder_raw)/x for x in ['train/train.csv','test/test.csv','breed_labels.csv',
            'color_labels.csv','state_labels.csv','train_images','test_images'])
        if not a.stanford_raw:required.append(reference/'stanford_dogs/joint_duplicate_split_guard.csv')
    if a.pawpularity_raw:
        required.extend([Path(a.pawpularity_raw)/'train.csv',Path(a.pawpularity_raw)/'train'])
        if not a.petfinder_raw:required.append(reference/'petfinder_images/image_manifest.csv')
        if not a.stanford_raw:required.append(reference/'stanford_dogs/joint_duplicate_split_guard.csv')
    if not a.skip_austin:
        for ii,oo in [('wter-evkm','9t4d-g238'),('pyqf-r2dc','gsvs-ypi7')]:
            pair=[data/f'{x}.json' for x in [ii,oo]]
            if any(x.exists() for x in pair):required.extend(pair)
    missing=sorted({str(x) for x in required if not x.exists()})
    if missing:raise SystemExit('Missing requested input/dependency:\n'+'\n'.join(missing))
    if out.exists():raise SystemExit('Choose a new output directory to preserve frozen results.')
    out.mkdir(parents=True);status={}
    def run(name,script,args):
        command=[sys.executable,str(root/script)]+[str(x) for x in args]+['--out',str(out/name)]
        status[name]={'execution':'running'}
        try:
            subprocess.run(command,check=True)
        except Exception:
            status[name]={'execution':'failed'}
            raise
        status[name]={'execution':'completed'}
    try:
        for era,intakes,outcomes,cutoff,v,t in [
            ('legacy','wter-evkm','9t4d-g238','2025-05-05T00:00:00-05:00','2023-01-01T00:00:00-06:00','2024-01-01T00:00:00-06:00'),
            ('current','pyqf-r2dc','gsvs-ypi7','2026-10-11T19:00:00-05:00','2026-01-01T00:00:00-06:00','2026-07-01T00:00:00-05:00')]:
            if a.skip_austin:
                status['austin_'+era]={'execution':'not_requested'}
                continue
            if not (data/f'{intakes}.json').is_file():
                status['austin_'+era]={'execution':'missing_input','required':str(data)}
                continue
            run('austin_'+era,'furcast_data.py',['austin','--era',era,'--intakes',data/f'{intakes}.json',
                '--outcomes',data/f'{outcomes}.json','--cutoff',cutoff,'--validation-start',v,'--test-start',t])
            run('austin_'+era+'_arrays','export_austin_arrays.py',['--input',out/('austin_'+era)])
        if a.oxford_images:
            run('oxford','audit_oxford_images.py',['--images',Path(a.oxford_images).resolve(),'--annotations',data/'oxford_annotations.tar.gz'])
        else:
            if (data/'oxford_annotations.tar.gz').is_file():
                run('oxford_annotations','furcast_data.py',['oxford','--annotations',data/'oxford_annotations.tar.gz'])
            status['oxford_images']={'execution':'missing_input','required':'--oxford-images path to separately supplied original image archive'}
        if a.stanford_raw:
            oxford_manifest=out/'oxford/image_splits.csv' if a.oxford_images else reference/'oxford/image_splits.csv'
            run('stanford_dogs','prepare_stanford_dogs.py',['--raw',Path(a.stanford_raw).resolve(),
                '--oxford-manifest',oxford_manifest,'--export-zip',out/'Stanford_Dogs_224_Preprocessed.zip'])
            status['stanford_dogs']['oxford_hash_source']=str(oxford_manifest)
        else:status['stanford_dogs']={'execution':'not_requested','required':'--stanford-raw directory; original images.tar is separate from starter ZIP'}
        if a.petfinder_raw:
            if a.petfinder_csv:raise ValueError('Choose --petfinder-raw OR provisional --petfinder-csv')
            raw=Path(a.petfinder_raw).resolve()
            run('petfinder_images','audit_petfinder_images.py',['--raw',raw])
            run('petfinder_duplicates','find_petfinder_duplicates.py',[
                '--petfinder-manifest',out/'petfinder_images/image_manifest.csv',
                '--breed-manifest',(out if a.stanford_raw else reference)/'stanford_dogs/joint_duplicate_split_guard.csv'])
            run('petfinder_adoption','prepare_petfinder_real.py',['--raw',raw,
                '--images',out/'petfinder_images/image_manifest.csv',
                '--edges',out/'petfinder_duplicates/petfinder_image_candidates.csv',
                '--export-images',out/'PetFinder_Default_Photos_224.zip'])
            status['petfinder_adoption']['scope']='Real files processed; photo/field time provenance and human identity review remain unverified'
        if a.pawpularity_raw:
            pf_manifest=out/'petfinder_images/image_manifest.csv' if a.petfinder_raw else reference/'petfinder_images/image_manifest.csv'
            breed_manifest=out/'stanford_dogs/joint_duplicate_split_guard.csv' if a.stanford_raw else reference/'stanford_dogs/joint_duplicate_split_guard.csv'
            run('pawpularity','prepare_pawpularity.py',['--raw',Path(a.pawpularity_raw).resolve(),
                '--external-manifest',pf_manifest,'--external-manifest',breed_manifest,
                '--export-images',out/'Pawpularity_Photos_300.zip'])
            status['pawpularity']['scope']='Supplied local files processed; only image-component independence, no rescuer/time claims'
        for name,path in [('petfinder',a.petfinder_csv),('pawpularity',a.pawpularity_csv)]:
            if path:
                args=[name,'--csv',Path(path).resolve()]
                if name=='petfinder' and a.reviewed_edges:args+=['--edges',Path(a.reviewed_edges).resolve()]
                run(name,'furcast_data.py',args)
                status[name]['scope']='CSV validation only; image audit and feature-time provenance must be reviewed separately'
            elif not ((name=='petfinder' and a.petfinder_raw) or (name=='pawpularity' and a.pawpularity_raw)):
                status[name]={'execution':'missing_input','required':'authorized original competition files'}
    finally:
        (out/'run_status.json').write_text(json.dumps({'version':'0.5.0','components':status,
            'completed_stages':sum(s['execution']=='completed' for s in status.values()),
            'all_components_complete':bool(status) and all(s['execution']=='completed' for s in status.values()),
            'scope':'Preprocessing execution only. Completed execution does not certify causal validity, prospective feature availability or production readiness.'},indent=2))
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
