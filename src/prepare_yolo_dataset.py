"""Build a NEW, sequence-disjoint dataset; never move or overwrite existing data."""
import argparse
import os
import random
import shutil
from pathlib import Path
import yaml
from PIL import Image
from dataset_utils import NAMES, resolve, read_yaml, read_mot, image_map, yolo_lines, file_hash, validate_root, write_json


def split_sequences(names, train_ratio=.85, seed=42):
    if len(set(names)) != len(names) or len(names) < 2 or not 0 < train_ratio < 1:
        raise ValueError('Need two or more unique sequences and a ratio between zero and one')
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = max(1, min(len(names)-1, round(len(names)*train_ratio)))
    return set(names[:n]), set(names[n:])


def prepare(train_root, test_root, output, seed=42, train_ratio=.85, allow_missing=False, link_images=False):
    train_root, output = Path(train_root).resolve(), Path(output).resolve()
    test_root = Path(test_root).resolve() if test_root else None
    for raw in (train_root,test_root):
        if raw is None:
            continue
        if not (raw/'images').is_dir() or not (raw/'annotations').is_dir():
            raise FileNotFoundError(f'Expected images/ and annotations/ under {raw}')
        if output == raw or output in raw.parents or raw in output.parents:
            raise ValueError('Output must be separate from raw data')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new directory')
    train_csvs = sorted((train_root/'annotations').glob('*.csv'))
    test_csvs = sorted((test_root/'annotations').glob('*.csv')) if test_root else []
    if not train_csvs or (test_root and not test_csvs):
        raise ValueError('No annotation CSV files found')
    training, validation = split_sequences([p.stem for p in train_csvs],train_ratio,seed)
    if (training|validation)&{p.stem for p in test_csvs}:
        raise ValueError('TrainReal and TestReal sequence IDs overlap')
    splits = ('train','val','test') if test_root else ('train','val')
    for split in splits:
        for kind in ('images','labels'):
            (output/kind/split).mkdir(parents=True,exist_ok=True)
    audit = {'seed':seed,'train_ratio':train_ratio,'class_names':NAMES,
             'scope':'Annotated frames only; unannotated images are not assumed to be verified empty scenes.',
             'raw_roots':{'train':str(train_root),'test':str(test_root) if test_root else None},
             'annotation_sha256':{},'missing_annotated_frames':[],'unannotated_images':0,
             'copy_method':'hardlink-with-copy-fallback' if link_images else 'copy'}
    for raw, paths in ((train_root,train_csvs),(test_root,test_csvs)):
        for csv_path in paths:
            seq = csv_path.stem
            split = 'test' if raw == test_root else ('train' if seq in training else 'val')
            folder = raw/'images'/seq
            if not folder.is_dir():
                raise FileNotFoundError(f'No images for {csv_path}')
            frames, annotations = image_map(folder), read_mot(csv_path)
            audit['annotation_sha256'][f'{split}/{seq}'] = file_hash(csv_path)
            missing = sorted(set(annotations)-set(frames))
            if missing and not allow_missing:
                raise ValueError(f'{seq}: {len(missing)} annotated frames missing; inspect and explicitly use --allow-missing-frames if appropriate')
            audit['missing_annotated_frames'].extend({'sequence':seq,'frame':f,'split':split} for f in missing)
            audit['unannotated_images'] += len(set(frames)-set(annotations))
            exported = 0
            for frame in sorted(set(annotations)&set(frames)):
                source = frames[frame]
                with Image.open(source) as im:
                    width,height = im.size
                    im.verify()
                try:
                    lines = yolo_lines(annotations[frame],width,height)
                except ValueError as exc:
                    raise ValueError(f'{seq}, frame {frame}: {exc}') from exc
                stem = f'{seq}_{frame:010d}'
                dest = output/'images'/split/f'{stem}{source.suffix.lower()}'
                if link_images:
                    try:
                        os.link(source,dest)
                    except OSError:
                        shutil.copy2(source,dest)
                else:
                    shutil.copy2(source,dest)
                (output/'labels'/split/f'{stem}.txt').write_text('\n'.join(lines),encoding='utf-8')
                exported += 1
            print(f'{split}: {seq}: {exported} annotated frames',flush=True)
    audit['splits'] = validate_root(output,splits,verify_images=False)
    write_json(output/'manifest.json',audit)
    cfg = {'path':'.','train':'images/train','val':'images/val','names':NAMES}
    if test_root:
        cfg['test'] = 'images/test'
    (output/'dataset.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    for split in splits:
        (output/f'{split}_sequences.txt').write_text('\n'.join(audit['splits'][split]['sequences']),encoding='utf-8')
    print(f'Validated configuration: {output/"dataset.yaml"}',flush=True)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/dataset.yml')
    p.add_argument('--train-root',help='TrainReal folder containing images/ and annotations/')
    p.add_argument('--test-root',help='TestReal folder containing images/ and annotations/')
    p.add_argument('--output',default='data/processed_v2')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-ratio',type=float,default=.85)
    p.add_argument('--skip-test',action='store_true')
    p.add_argument('--allow-missing-frames',action='store_true')
    p.add_argument('--link-images',action='store_true',help='Hardlink images when possible; never edit linked image files')
    args = p.parse_args()
    cfg = read_yaml(resolve(args.config))
    train = resolve(args.train_root) if args.train_root else resolve(cfg['paths']['train']['images']).parent
    test = None if args.skip_test else (resolve(args.test_root) if args.test_root else resolve(cfg['paths']['test']['images']).parent)
    prepare(train,test,resolve(args.output),args.seed,args.train_ratio,args.allow_missing_frames,args.link_images)


if __name__ == '__main__':
    main()
