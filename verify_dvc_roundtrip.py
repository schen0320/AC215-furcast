"""Isolated SYNTHETIC DVC integration check; never creates release evidence.

Exercises extraction -> preprocessing -> validation, forced determinism,
local push -> empty-cache clone pull -> independently forced validation.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from furcast_data import PAW
from prepare_pawpularity import inspect_image, sha

ROOT = Path(__file__).resolve().parent
CODE = ['pawpularity_inputs.py','prepare_pawpularity.py','furcast_data.py',
        'audit_oxford_images.py','find_petfinder_duplicates.py',
        'validate_pawpularity_outputs.py','requirements-dvc.lock.txt',
        'dvc.yaml','params.yaml','.gitignore']


def main():
    env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
               DVC_NO_ANALYTICS='true')
    with tempfile.TemporaryDirectory(prefix='furcast-dvc-test-') as tmp:
        root=Path(tmp); repo=root/'producer'; repo.mkdir(); remote=root/'remote'
        def run(*args, cwd=repo):
            r=subprocess.run(args,cwd=cwd,env=env,text=True,capture_output=True)
            if r.returncode:
                raise RuntimeError(f'{args[0:3]} failed:\n{r.stdout}\n{r.stderr}')
            return r.stdout
        for name in CODE: shutil.copyfile(ROOT/name,repo/name)
        (repo/'docs').mkdir(); shutil.copyfile(ROOT/'docs/TARGETS_AND_SPLITS.md',repo/'docs/TARGETS_AND_SPLITS.md')
        run('git','init','-q'); run('dvc','init','-q'); run('dvc','config','core.analytics','false')
        raw=root/'fixture'; (raw/'train').mkdir(parents=True); (raw/'test').mkdir()
        rng=np.random.default_rng(215); rows=[]
        for i in range(25):
            ident=f'{i:032x}'
            Image.fromarray(rng.integers(0,256,(48,64,3),dtype=np.uint8)).save(raw/'train'/f'{ident}.jpg')
            rows.append(dict(Id=ident,Pawpularity=i+1,**{c:0 for c in PAW}))
        pd.DataFrame(rows).to_csv(raw/'train.csv',index=False)
        ident='f'*32
        Image.fromarray(rng.integers(0,256,(48,64,3),dtype=np.uint8)).save(raw/'test'/f'{ident}.jpg')
        pd.DataFrame([dict(Id=ident,**{c:0 for c in PAW})]).to_csv(raw/'test.csv',index=False)
        pd.DataFrame([dict(Id=ident,Pawpularity=50)]).to_csv(raw/'sample_submission.csv',index=False)
        refs=repo/'artifacts/reference'; refs.mkdir(parents=True)
        for i,name in enumerate(['breed_fingerprints.csv','petfinder_fingerprints.csv']):
            row=inspect_image((str(raw),'train',f'{i:032x}')); row['key']=f'external:{i}'
            pd.DataFrame([row]).to_csv(refs/name,index=False)
        archive=repo/'artifacts/inputs/pawpularity.zip'; archive.parent.mkdir()
        with ZipFile(archive,'w') as z:
            for p in sorted(raw.rglob('*')):
                if p.is_file(): z.write(p,p.relative_to(raw).as_posix())
        contract={'version':'SYNTHETIC_TEST_ONLY','archive_sha256':sha(archive),
                  'archive_bytes':archive.stat().st_size,'train_csv_sha256':sha(raw/'train.csv'),
                  'train_rows':25,'example_test_rows':1,
                  'reference_sha256':{p.name:sha(p) for p in refs.iterdir()}}
        (repo/'config').mkdir(); (repo/'config/pawpularity-source.json').write_text(json.dumps(contract))
        (repo/'params.yaml').write_text('pawpularity:\n  image_size: 300\n  workers: 1\n')
        run('dvc','add','artifacts/inputs/pawpularity.zip','artifacts/reference')
        run('dvc','repro','validate_pawpularity')
        lock=(repo/'dvc.lock').read_bytes()
        run('dvc','repro','validate_pawpularity')
        if lock!=(repo/'dvc.lock').read_bytes(): raise AssertionError('No-op changed lock')
        run('dvc','repro','--force','validate_pawpularity')
        if lock!=(repo/'dvc.lock').read_bytes(): raise AssertionError('Forced rebuild changed lock')
        for stage in ['extract_pawpularity','prepare_pawpularity','validate_pawpularity']:
            if json.loads(run('dvc','status','--json',stage)): raise AssertionError('DVC stage not clean')
        # A source byte change must invalidate provenance before preprocessing.
        original=(refs/'breed_fingerprints.csv').read_bytes()
        (refs/'breed_fingerprints.csv').write_bytes(original+b'\n')
        invalid=subprocess.run([sys.executable,'pawpularity_inputs.py','check','--raw','artifacts/raw/pawpularity',
                                '--reference','artifacts/reference'],cwd=repo,env=env,capture_output=True)
        if invalid.returncode==0: raise AssertionError('Changed reference accepted')
        (refs/'breed_fingerprints.csv').write_bytes(original)
        run('dvc','remote','add','--local','-d','roundtrip',str(remote))
        targets=['artifacts/inputs/pawpularity.zip.dvc','artifacts/reference.dvc',
                 'extract_pawpularity','prepare_pawpularity','validate_pawpularity']
        run('dvc','push',*targets)
        run('git','add','.'); run('git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Synthetic integration fixture')
        consumer=root/'consumer'; run('git','clone','-q',str(repo),str(consumer))
        run('dvc','remote','add','--local','-d','roundtrip',str(remote),cwd=consumer)
        run('dvc','pull',*targets,cwd=consumer)
        expected=json.loads((repo/'artifacts/validation/rebuild.json').read_text())
        run(sys.executable,'validate_pawpularity_outputs.py','--out','artifacts/processed/pawpularity',
            '--photos','artifacts/processed/Pawpularity_Photos_300.zip','--report','artifacts/validation/recheck.json',cwd=consumer)
        actual=json.loads((consumer/'artifacts/validation/recheck.json').read_text())
        if actual!=expected: raise AssertionError('Consumer validation changed')
        print(json.dumps({'status':'passed','data':'synthetic_only','raw_rows':25,
            'retained_rows':actual['validated_prepared_images'],'forced_rebuild_identical_lock':True,
            'no_op_identical_lock':True,'changed_reference_rejected':True,
            'local_push_empty_cache_pull_validated':True,'real_raw_rebuild_proven':False},indent=2))


if __name__=='__main__': main()
