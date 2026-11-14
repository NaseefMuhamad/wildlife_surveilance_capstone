"""Read-only spot audit of exported labels against the original same-frame CSV rows."""
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image
from dataset_utils import resolve, image_map, read_mot, yolo_lines, file_hash, sequence_id, write_json


def audit(dataset, raw, sample_per_sequence=3):
    dataset,raw=Path(dataset),Path(raw)
    grouped={}
    for split in ('train','val'):
        groups=defaultdict(list)
        for path in sorted((dataset/'labels'/split).glob('*.txt')):
            groups[sequence_id(path.stem)].append(path)
        if not groups:
            raise ValueError(f'No labels for {split}')
        grouped[split]=groups
    checks=[]
    for seq in sorted(set(grouped['train'])|set(grouped['val'])):
        original=read_mot(raw/'annotations'/f'{seq}.csv')
        images=image_map(raw/'images'/seq)
        for split in ('train','val'):
            labels=grouped[split].get(seq,[])
            if not labels:
                continue
            indices=sorted({round(i*(len(labels)-1)/max(sample_per_sequence-1,1)) for i in range(sample_per_sequence)})
            for i in indices:
                path=labels[i]; frame=int(path.stem.rsplit('_',1)[-1])
                exported=dataset/'images'/split/f'{path.stem}.jpg'
                with Image.open(exported) as im:
                    width,height=im.size
                expected=sorted(tuple(map(float,line.split())) for line in yolo_lines(original.get(frame,[]),width,height))
                actual=sorted(tuple(map(float,line.split())) for line in path.read_text().splitlines())
                match=len(actual)==len(expected) and all(len(a)==len(b)==5 and all(abs(x-y)<=1.1e-6 for x,y in zip(a,b)) for a,b in zip(actual,expected))
                checks.append({'split':split,'stem':path.stem,'label_matches_same_frame_csv':match,
                               'image_matches_raw':frame in images and file_hash(exported)==file_hash(images[frame]),
                               'expected_boxes':len(expected),'exported_boxes':len(actual)})
        print(f'Audited sequence {seq}',flush=True)
    return {'dataset':str(dataset),'raw':str(raw),'samples':len(checks),
            'training_sequences':len(grouped['train']),'validation_sequences':len(grouped['val']),
            'overlapping_sequences':sorted(set(grouped['train'])&set(grouped['val'])),
            'mismatches':[c for c in checks if not c['label_matches_same_frame_csv'] or not c['image_matches_raw']],
            'checks':checks,'scope':'Deterministic sampled audit; not a full annotation-alignment certification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/processed')
    p.add_argument('--raw-root',default='data/raw/TrainReal/TrainReal')
    p.add_argument('--samples-per-sequence',type=int,default=3)
    p.add_argument('--output',default='outputs/label_mapping_audit.json')
    args=p.parse_args()
    if args.samples_per_sequence < 1:
        p.error('Need at least one sample')
    result=audit(resolve(args.dataset),resolve(args.raw_root),args.samples_per_sequence)
    write_json(resolve(args.output),result)
    print(f"Checked {result['samples']} samples; mismatches={len(result['mismatches'])}; overlapping sequences={len(result['overlapping_sequences'])}")

if __name__ == '__main__':
    main()
