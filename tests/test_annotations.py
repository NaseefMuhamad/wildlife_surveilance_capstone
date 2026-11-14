import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image
from dataset_utils import validate_root, read_mot, image_map, dataset_config
from prepare_yolo_dataset import prepare, split_sequences


def raw_fixture(root, names):
    (root/'annotations').mkdir(parents=True)
    for name in names:
        folder=root/'images'/name
        folder.mkdir(parents=True)
        # Gapped frame IDs catch the original list-indexing bug.
        Image.new('RGB',(100,80),'black').save(folder/f'{name}_0000000010.jpg')
        Image.new('RGB',(100,80),'white').save(folder/f'{name}_0000000030.jpg')
        (root/'annotations'/f'{name}.csv').write_text(
            '10,1,10,8,20,16,0,1,0,0\n30,2,50,40,10,8,1,0,0,0\n')


class DatasetRegressionTests(unittest.TestCase):
    def test_no_splits_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                validate_root(temp,())
            config=Path(temp)/'no_splits.yaml'
            config.write_text(f'path: {Path(temp).as_posix()}\nnames:\n  0: animal\n  1: human\n')
            with self.assertRaises(ValueError):
                dataset_config(config)

    def test_documented_27_5_split_is_deterministic_and_disjoint(self):
        names=[f'{i:010d}_0000000000' for i in range(32)]
        train,val=split_sequences(names)
        self.assertEqual((len(train),len(val)),(27,5))
        self.assertFalse(train&val)
        self.assertEqual((train,val),split_sequences(list(reversed(names))))

    def test_frame_ids_are_matched_and_source_data_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); train=root/'rawtrain'; test=root/'rawtest'; out=root/'new'
            raw_fixture(train,['a_0000','b_0000']); raw_fixture(test,['c_0000'])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest=prepare(train,test,out)
            self.assertEqual(manifest['splits']['test']['images'],2)
            line=(out/'labels/test/c_0000_0000000010.txt').read_text()
            self.assertEqual(list(map(float,line.split())),[0.,.2,.2,.2,.2])
            self.assertTrue((train/'images/a_0000/a_0000_0000000010.jpg').exists())
            validate_root(out)
            with self.assertRaises(FileExistsError):
                prepare(train,test,out)

    def test_missing_and_empty_datasets_do_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                validate_root(Path(temp)/'missing')
            for split in ('train','val'):
                (Path(temp)/'images'/split).mkdir(parents=True)
                (Path(temp)/'labels'/split).mkdir(parents=True)
            with self.assertRaises(ValueError):
                validate_root(temp,('train','val'))

    def test_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split in ('train','val'):
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(100,80)).save(root/'images'/split/'same_sequence_0000000010.jpg')
                (root/'labels'/split/'same_sequence_0000000010.txt').write_text('0 .2 .2 .1 .1\n1 .5 .5 .1 .1')
            with self.assertRaisesRegex(ValueError,'Sequence leakage'):
                validate_root(root,('train','val'))

    def test_invalid_boxes_classes_and_pairing_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'images/test').mkdir(parents=True); (root/'labels/test').mkdir(parents=True)
            Image.new('RGB',(100,80)).save(root/'images/test/one_sequence_0000000010.jpg')
            label=root/'labels/test/one_sequence_0000000010.txt'
            for bad in ('0 .9 .5 .8 .2','0 .5 .5 0 .2','0 nan .5 .2 .2','2 .5 .5 .2 .2','0 .5 .5 .2'):
                label.write_text(bad+'\n1 .5 .5 .1 .1')
                with self.assertRaises(ValueError):
                    validate_root(root,('test',))
            label.unlink()
            with self.assertRaises(ValueError):
                validate_root(root,('test',))

    def test_missing_annotated_frames_require_explicit_acknowledgment(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); raw_fixture(root/'raw',['a_0000','b_0000'])
            path=root/'raw/annotations/a_0000.csv'
            path.write_text(path.read_text()+'99,3,1,1,2,2,1,0,0,0\n')
            with self.assertRaisesRegex(ValueError,'annotated frames missing'):
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(root/'raw',None,root/'strict')
            with contextlib.redirect_stdout(io.StringIO()):
                manifest=prepare(root/'raw',None,root/'acknowledged',allow_missing=True)
            self.assertEqual(manifest['missing_annotated_frames'][0]['frame'],99)

    def test_manifest_detects_modified_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); raw_fixture(root/'raw',['a_0000','b_0000'])
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root/'raw',None,root/'dataset')
            label=next((root/'dataset/labels/train').glob('*.txt'))
            label.write_text(label.read_text().replace('.20000000','.25000000'))
            with self.assertRaisesRegex(ValueError,'manifest'):
                validate_root(root/'dataset',('train','val'))

    def test_nonfinite_annotations_and_duplicate_frame_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); root.joinpath('bad.csv').write_text('10,1,nan,1,2,2,0,1,0,0\n')
            with self.assertRaises(ValueError):
                read_mot(root/'bad.csv')
            Image.new('RGB',(10,10)).save(root/'a_10.jpg')
            Image.new('RGB',(10,10)).save(root/'b_10.png')
            with self.assertRaises(ValueError):
                image_map(root)


if __name__ == '__main__':
    unittest.main()
