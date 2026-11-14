"""Validate the dataset selected by the same YAML used for training/evaluation."""
import argparse
import json
from dataset_utils import dataset_config, validate_root

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='configs/yolo_dataset.yaml')
    args=p.parse_args()
    root,cfg=dataset_config(args.data)
    report=validate_root(root,tuple(s for s in ('train','val','test') if s in cfg))
    print(json.dumps(report,indent=2))
    print('All structural, manifest and sequence-overlap checks passed.')

if __name__ == '__main__':
    main()
