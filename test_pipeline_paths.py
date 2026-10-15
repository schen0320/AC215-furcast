"""Repository handoff regressions. Fixtures are not acquired source data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_data_pipeline import main


def invoke(args):
    with patch('sys.argv', ['run_data_pipeline.py', *map(str, args)]):
        main()


class PipelinePaths(unittest.TestCase):
    def test_empty_clone_cannot_claim_completed_data(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            invoke(['--data-root', root/'raw', '--reference-results', root/'prior', '--out', root/'out'])
            status = json.loads((root/'out/run_status.json').read_text())
            self.assertEqual(status['completed_stages'], 0)
            self.assertFalse(status['all_components_complete'])
            self.assertEqual(status['components']['pawpularity']['execution'], 'missing_input')

    def test_partial_austin_snapshot_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'wter-evkm.json').write_text('[]')
            with self.assertRaises(SystemExit):
                invoke(['--data-root', root, '--out', root/'out'])
            self.assertFalse((root/'out').exists())

    def test_paw_cannot_skip_missing_cross_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'train').mkdir(); (root/'train.csv').write_text('Id,Pawpularity\n')
            with self.assertRaises(SystemExit):
                invoke(['--skip-austin', '--data-root', root/'absent', '--pawpularity-raw', root,
                        '--reference-results', root/'absent', '--out', root/'out'])
            self.assertFalse((root/'out').exists())

    def test_joint_run_uses_new_breed_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory); data = root/'data'; data.mkdir()
            for name in ('oxford_annotations.tar.gz', 'oxford_images.tar.gz'):
                (data/name).touch()
            stanford = root/'stanford'; stanford.mkdir()
            for name in ('images.tar', 'annotation.tar', 'lists.tar', 'download_manifest.json'):
                (stanford/name).touch()
            pf = root/'pf'; pf.mkdir()
            for name in ('train', 'test', 'train_images', 'test_images'):
                (pf/name).mkdir()
            for name in ('train/train.csv', 'test/test.csv', 'breed_labels.csv', 'color_labels.csv', 'state_labels.csv'):
                (pf/name).touch()
            with patch('run_data_pipeline.subprocess.run') as run:
                invoke(['--skip-austin', '--data-root', data, '--out', root/'out',
                        '--oxford-images', data/'oxford_images.tar.gz', '--stanford-raw', stanford,
                        '--petfinder-raw', pf])
            command = next(c.args[0] for c in run.call_args_list if c.args[0][1].endswith('find_petfinder_duplicates.py'))
            self.assertEqual(command[command.index('--breed-manifest')+1],
                             str(root/'out/stanford_dogs/joint_duplicate_split_guard.csv'))


if __name__ == '__main__':
    unittest.main()
