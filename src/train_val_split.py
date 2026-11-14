"""Retired: never shuffle or move individual frames between splits."""
if __name__ == '__main__':
    raise SystemExit('Frame-level splitting is disabled. Run python src/prepare_yolo_dataset.py to build a NEW sequence-disjoint dataset. Existing files are preserved.')
