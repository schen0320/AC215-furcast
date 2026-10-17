"""Acquisition guards use synthetic archives, never release evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from pawpularity_inputs import check_hash, checked_members, extract, import_delivery


class InputGuards(unittest.TestCase):
    def test_missing_or_changed_input_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'missing.zip'
            with self.assertRaises(FileNotFoundError): check_hash(p, '0'*64)
            p.write_bytes(b'not the release')
            with self.assertRaisesRegex(ValueError, 'SHA-256'): check_hash(p, '0'*64)

    def test_archive_traversal_duplicate_symlink_stops(self):
        from zipfile import ZipInfo
        import warnings
        for kind in ['traversal','duplicate','symlink']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                p=Path(d)/'bad.zip'
                with ZipFile(p,'w') as z, warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    if kind=='traversal': z.writestr('../escape.txt',b'bad')
                    elif kind=='duplicate':
                        z.writestr('a.txt',b'one'); z.writestr('a.txt',b'two')
                    else:
                        i=ZipInfo('link'); i.create_system=3; i.external_attr=0o120777<<16
                        z.writestr(i,b'../../target')
                with ZipFile(p) as z, self.assertRaises(ValueError): checked_members(z)

    def test_wrong_package_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.zip'; p.write_bytes(b'fake'); out=Path(d)/'out'
            with self.assertRaises(ValueError):
                import_delivery(p,out,{'processed_package_sha256':'0'*64})
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                extract(p,out,{'archive_sha256':'0'*64})
            self.assertFalse(out.exists())

    def test_optimized_python_cannot_skip_output_validation(self):
        import subprocess, sys
        result=subprocess.run([sys.executable,'-O','validate_pawpularity_outputs.py',
                               '--out','missing','--photos','missing.zip'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires assertions',result.stderr)


if __name__=='__main__': unittest.main()
