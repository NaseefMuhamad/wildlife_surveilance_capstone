"""Dataset paths, MOT frame matching and fail-closed validation; no ML imports."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
NAMES = {0: 'animal', 1: 'human'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def resolve(path, base=PROJECT):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base)/path).resolve()


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sequence_id(stem):
    seq, frame = stem.rsplit('_', 1)
    if not frame.isdigit():
        raise ValueError(f'Expected sequence_frame filename: {stem}')
    return seq


def image_map(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        frame = int(path.stem.rsplit('_', 1)[-1])
        if frame in result:
            raise ValueError(f'Duplicate frame ID {frame} in {folder}')
        result[frame] = path
    return result


def read_mot(path):
    frames = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        for number, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if len(row) != 10:
                raise ValueError(f'{path}:{number}: expected ten MOT fields')
            values = list(map(float, row))
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'{path}:{number}: non-finite annotation')
            frame, cls = values[0], values[6]
            if frame != int(frame) or frame < 0 or cls != int(cls) or int(cls) not in NAMES:
                raise ValueError(f'{path}:{number}: invalid frame or class')
            frames.setdefault(int(frame), []).append(values)
    if not frames:
        raise ValueError(f'No annotations: {path}')
    return frames


def yolo_lines(rows, width, height):
    lines = []
    for row in rows:
        x, y, w, h = row[2:6]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > width or y+h > height:
            raise ValueError(f'Invalid box {(x,y,w,h)} for {width}x{height}')
        lines.append(f'{int(row[6])} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}')
    return lines


def label_hash(folder):
    h = hashlib.sha256()
    for path in sorted(Path(folder).glob('*.txt')):
        h.update(path.name.encode()+b'\0'+path.read_bytes()+b'\0')
    return h.hexdigest()


def validate_root(root, splits=('train', 'val', 'test'), verify_images=True):
    if not splits or len(set(splits)) != len(splits) or any(s not in ('train','val','test') for s in splits):
        raise ValueError('Specify at least one distinct train/val/test split')
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    report, sequences = {}, {}
    for split in splits:
        image_dir, label_dir = root/'images'/split, root/'labels'/split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Missing image/label directories for {split}')
        files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        images = {p.stem:p for p in files}
        labels = {p.stem:p for p in label_dir.glob('*.txt')}
        if len(images) != len(files):
            raise ValueError(f'{split}: duplicate image stems')
        if not images or not labels:
            raise ValueError(f'{split}: no image/label pairs')
        if images.keys() != labels.keys():
            raise ValueError(f'{split}: image/label pairing mismatch')
        sequences[split] = {sequence_id(stem) for stem in images}
        counts = Counter()
        for stem, label in labels.items():
            if verify_images:
                with Image.open(images[stem]) as im:
                    im.verify()
            content = label.read_text(encoding='utf-8').strip()
            if not content:
                counts['empty_labels'] += 1  # permitted for verified empty scenes
            for number, line in enumerate(content.splitlines(), 1):
                if len(line.split()) != 5:
                    raise ValueError(f'{label}:{number}: expected five YOLO fields')
                c, x, y, w, h = map(float, line.split())
                if not all(math.isfinite(v) for v in (c,x,y,w,h)):
                    raise ValueError(f'{label}:{number}: non-finite value')
                if c != int(c) or int(c) not in NAMES:
                    raise ValueError(f'{label}:{number}: invalid class')
                if w <= 0 or h <= 0 or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
                    raise ValueError(f'{label}:{number}: box outside image or nonpositive size')
                counts[NAMES[int(c)]] += 1
        missing = set(NAMES.values())-counts.keys()
        if missing:
            raise ValueError(f'{split}: no annotations for {sorted(missing)}')
        report[split] = {'images':len(images), 'sequences':sorted(sequences[split]), 'boxes':dict(counts), 'label_sha256':label_hash(label_dir)}
    for i, left in enumerate(splits):
        for right in splits[i+1:]:
            overlap = sequences[left]&sequences[right]
            if overlap:
                raise ValueError(f'Sequence leakage between {left}/{right}: {sorted(overlap)}')
    manifest = root/'manifest.json'
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding='utf-8'))['splits']
        for split in splits:
            if split not in recorded or any(recorded[split][k] != report[split][k] for k in ('images','sequences','label_sha256')):
                raise ValueError(f'{split}: dataset no longer matches its manifest')
    return report


def dataset_config(path):
    config_path = resolve(path)
    cfg = read_yaml(config_path)
    dataset_path = cfg.get('path', 'data/processed_v2')
    # Exported YAML files use "." to refer to their own dataset directory.
    root = config_path.parent if dataset_path == '.' else resolve(dataset_path)
    names = cfg.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != NAMES:
        raise ValueError(f'Expected names={NAMES}, got {names}')
    if not any(s in cfg for s in ('train','val','test')):
        raise ValueError('Dataset YAML must define at least one train/val/test split')
    for split in ('train','val','test'):
        if split in cfg and cfg[split] != f'images/{split}':
            raise ValueError(f'Expected {split}: images/{split}')
    return root, cfg


def absolute_config(path):
    root, cfg = dataset_config(path)
    cfg['path'] = root.as_posix()
    folder = PROJECT/'outputs/runtime_configs'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/f'{hashlib.sha256(str(root).encode()).hexdigest()[:12]}.yaml'
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return output
