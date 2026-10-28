import yaml
import pandas as pd
import cv2

from pathlib import Path
from collections import defaultdict
from shutil import copy2


def load_config():
    with open("configs/dataset.yml", "r") as f:
        return yaml.safe_load(f)


def convert_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h

    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def main():

    cfg = load_config()

    ann_root = Path(cfg["paths"]["train"]["annotations"])
    img_root = Path(cfg["paths"]["train"]["images"])

    out_img_dir = Path("data/processed/images/train")
    out_lbl_dir = Path("data/processed/labels/train")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    total_images = 0

    for csv_file in ann_root.glob("*.csv"):

        seq_name = csv_file.stem

        sequence_folder = img_root / seq_name

        if not sequence_folder.exists():
            continue

        df = pd.read_csv(
            csv_file,
            header=None,
            names=cfg["annotation_columns"]
        )

        grouped = df.groupby("frame_number")

        images = sorted(sequence_folder.glob("*.jpg"))

        for frame_id, frame_df in grouped:

            if frame_id >= len(images):
                continue

            img_path = images[int(frame_id)]

            img = cv2.imread(str(img_path))

            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            image_name = img_path.stem

            label_path = out_lbl_dir / f"{image_name}.txt"

            yolo_lines = []

            for _, row in frame_df.iterrows():

                cls = int(row["class"])

                x = row["x"]
                y = row["y"]
                w = row["w"]
                h = row["h"]

                xc, yc, wn, hn = convert_bbox(
                    x, y, w, h,
                    img_w, img_h
                )

                yolo_lines.append(
                    f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                )

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

            copy2(
                img_path,
                out_img_dir / img_path.name
            )

            total_labels += len(yolo_lines)
            total_images += 1

    print("\nConversion completed")
    print(f"Images processed : {total_images}")
    print(f"Labels generated : {total_labels}")


if __name__ == "__main__":
    main()
