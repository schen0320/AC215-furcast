"""Synthetic fixtures only: no fixture is presented as real pet data."""
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
from furcast_data import PAW
from prepare_pawpularity import prepare, inspect_image, validate_labels, group_and_split


class PawGuards(unittest.TestCase):
    def test_transitive_cross_source_quarantine(self):
        # A--B--external bridge; A has no direct edge to external.
        rows=[dict(key='pawpularity:train/'+str(i),Id=str(i),source_partition='train',
                   decode_ok=True,phash=f'{h:016x}',pixel_sha256=str(i))
              for i,h in enumerate([0,63,0x123456789abcdef0,0xdeadbeef12345678,
                                     0xfedcba9876543210,0x87654321abcdef01,
                                     0x55aa55aa55aa55aa,0x1122334455667788,
                                     0xababcdcd13135757,0x78efba34cd568912])]
        external=pd.DataFrame([dict(key='petfinder:bridge',phash='0000000000000fff',pixel_sha256='outside')])
        result,_=group_and_split(pd.DataFrame(rows),external)
        self.assertTrue(result.iloc[:2].split.eq('excluded').all())
        self.assertTrue(result.iloc[:2].exclusion_reason.eq('cross_source_duplicate_candidate').all())
        self.assertEqual(result.loc[result.split.ne('excluded')].groupby('duplicate_component').split.nunique().max(),1)

    def test_missing_data_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            with self.assertRaises(FileNotFoundError):prepare(d,out,[])
            self.assertFalse(out.exists())

    def test_invalid_label_or_id_rejected(self):
        row=dict(Id='a'*32,Pawpularity=10,**{c:0 for c in PAW})
        for update in [dict(Id='../bad'),dict(Pawpularity=101),dict(Pawpularity=2.5),{'Eyes':2}]:
            with self.subTest(update=update),self.assertRaises(ValueError):
                validate_labels(pd.DataFrame([{**row,**update}]))

    def test_pipeline_excludes_examples_and_keeps_targets_out_of_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'raw';(raw/'train').mkdir(parents=True);(raw/'test').mkdir()
            rng=np.random.default_rng(215);rows=[]
            for i in range(25):
                ident=f'{i:032x}'
                Image.fromarray(rng.integers(0,256,size=(48,64,3),dtype=np.uint8)).save(raw/'train'/f'{ident}.jpg')
                rows.append(dict(Id=ident,Pawpularity=i+1,**{c:0 for c in PAW}))
            # Internal identical image with a different score: remains one split.
            (raw/'train'/f'{24:032x}.jpg').write_bytes((raw/'train'/f'{23:032x}.jpg').read_bytes())
            # Corrupt source must be excluded, not silently repaired.
            (raw/'train'/f'{22:032x}.jpg').write_bytes(b'not an image')
            rows[2]['Eyes']=1;rows[2]['Blur']=1
            pd.DataFrame(rows).to_csv(raw/'train.csv',index=False)
            ident='f'*32
            Image.fromarray(rng.integers(0,256,size=(48,64,3),dtype=np.uint8)).save(raw/'test'/f'{ident}.jpg')
            pd.DataFrame([dict(Id=ident,**{c:0 for c in PAW})]).to_csv(raw/'test.csv',index=False)
            ext=inspect_image((str(raw),'train',f'{0:032x}'));ext['key']='petfinder:existing_test'
            pd.DataFrame([ext]).to_csv(root/'external.csv',index=False)
            report=prepare(raw,root/'out',[root/'external.csv'],root/'photos.zip',workers=1)
            labels=pd.read_csv(root/'out/labels_and_splits.csv',dtype={'Id':str})
            inputs=pd.read_csv(root/'out/image_inputs.csv',dtype={'Id':str})
            self.assertEqual(report['source_train_rows'],25)
            self.assertEqual(report['retained_rows'],23)
            self.assertEqual(report['decode_failures'],1)
            self.assertEqual(report['blur_eyes_review_rows'],1)
            self.assertEqual(report['identical_pixel_groups_with_different_scores'],1)
            self.assertNotIn(ident,labels.Id.tolist())
            self.assertNotIn('Pawpularity',inputs.columns)
            self.assertFalse(set(PAW)&set(inputs.columns))
            self.assertEqual(labels.loc[labels.Id.isin([f'{23:032x}',f'{24:032x}'])].split.nunique(),1)
            self.assertEqual(labels.loc[labels.Id.eq(f'{0:032x}'),'split'].iloc[0],'excluded')
            with ZipFile(root/'photos.zip') as z:self.assertEqual(len(z.namelist()),23)
            # Removing a source photo must fail before producing a successful audit.
            (raw/'train'/f'{1:032x}.jpg').unlink()
            with self.assertRaisesRegex(ValueError,'missing=1'):
                prepare(raw,root/'out2',[root/'external.csv'],workers=1)


if __name__=='__main__':unittest.main()
