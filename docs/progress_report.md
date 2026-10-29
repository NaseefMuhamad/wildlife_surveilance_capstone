# Thermal Wildlife Surveillance for Ranger Decision Support

## Project Overview

The goal of this project is to develop a thermal wildlife surveillance system capable of detecting and monitoring humans and animals from aerial thermal imagery. The system aims to support ranger decision-making by automatically identifying objects of interest in thermal video streams and providing the foundation for future tracking and alert mechanisms.

The project utilizes the BIRDSAI thermal wildlife dataset and follows a six-week implementation roadmap beginning with dataset exploration and ending with the deployment of an integrated surveillance pipeline

---

# Project Roadmap

| Week | Focus Area | Expected Output |
|--------|------------|------------|
| Week 1 | Data + Preprocessing | Dataset audit, samples, splits |
| Week 2 | Baseline | YOLO results and failure analysis |
| Week 3 | Fine-Tuning | Metrics and error analysis |
| Week 4 | Controlled Improvements | Baseline vs improved comparison |
| Week 5 | Tracking + Robustness | Advanced CV results |
| Week 6 | Final Integration | Demo on unseen videos |

---

# Week 1: Data Exploration and Preprocessing

## Objective

The objective of Week 1 was to understand the structure, quality, and suitability of the BIRDSAI dataset before model development. Since the project focuses on aerial thermal imagery, it was necessary to understand how images and annotations were organized, what object classes were available, and whether the available data aligned with the surveillance objectives of this project.

---

## Activities Completed

The first activity involved exploring the dataset structure and understanding how data was organized.

The dataset was found to contain separate partitions for training and testing:

```text
TrainReal/
TestReal/
```

Within each partition, images were organized into thermal video sequences. Each sequence contains multiple image frames representing consecutive observations over time.

The annotation files were then inspected to determine the available information for detection and future tracking tasks.

Several sample sequences were visualized together with their annotations to verify correspondence between image frames and labeled objects.

---

## Annotation Analysis

The annotation files were found to follow the structure:

```text
frame_id,
object_id,
x,
y,
width,
height,
class,
species,
occlusion,
noise
```

The broad object classes available are:

```text
0 = Animal
1 = Human
```

Species labels available in the real dataset are:

| Species ID | Species |
|------------|------------|
| -1 | Unknown |
| 0 | Human |
| 1 | Elephant |
| 2 | Lion |
| 3 | Giraffe |

The presence of both broad object classes and species labels provides flexibility for either binary detection or species-specific detection approaches.

---

## Dataset Distribution Analysis

To better understand class balance, the annotation files were analyzed across all available training sequences.

### Broad Class Counts

| Class | Count |
|---------|---------:|
| Animal | 74,668 |
| Human | 12,531 |

### Species Counts

| Species | Count |
|---------|---------:|
| Elephant | 42,973 |
| Unknown | 20,692 |
| Human | 12,531 |
| Giraffe | 9,979 |
| Lion | 1,024 |

### Observations

The dataset is dominated by animal annotations, particularly elephants.

A substantial number of annotations belong to the unknown category, indicating instances where species-level identification was uncertain.

The lion class appears relatively underrepresented compared to other species, which may introduce challenges during model training.

---

## Challenges Encountered

Several challenges were encountered during the exploration phase.

The first challenge involved understanding the nested sequence-based folder structure used by the dataset.

The second challenge involved interpreting the meaning of each annotation field and determining how they relate to object detection and tracking tasks.

Another challenge involved deciding whether the project should focus on binary detection (animal versus human) or species-level detection.

---

## Week 1 Outcome

By the end of Week 1, the dataset structure, annotation schema, species mapping, and class distribution had been fully understood. This established a strong foundation for implementing a baseline object detection model in Week 2.

---

# Week 2: Baseline Model Implementation

## Objective

The objective of Week 2 was to establish a baseline object detection model using a pretrained YOLOv8 network and evaluate how well a model trained on conventional RGB imagery performs when applied directly to thermal wildlife imagery.

The baseline serves as a reference point against which future fine-tuned models will be compared.

---

## Activities Completed

The Ultralytics YOLO framework was installed and configured within the project environment.

A pretrained YOLOv8n model was downloaded and verified successfully.

Several image sequences from the BIRDSAI training dataset were selected for inference experiments in order to observe the model's behavior on thermal imagery.

Predictions were generated and reviewed to identify both strengths and weaknesses of the baseline model.

---

## Baseline Configuration

### Detection Model

```text
YOLOv8n
```

### Training Dataset

```text
COCO Dataset
```

### Evaluation Dataset

```text
BIRDSAI Thermal Wildlife Dataset
```

---

## Baseline Evaluation

### Human Detection Experiment

A sequence containing two human targets was selected for evaluation.

#### Ground Truth

```text
2 Humans
```

#### YOLO Prediction

```text
2 Persons
```

#### Observation

The model successfully localized both targets and correctly classified them as persons.

This demonstrated that features learned from RGB imagery can still transfer reasonably well to thermal human detection.

---

### Wildlife Detection Experiment

A sequence containing elephant targets was selected for evaluation.

#### Ground Truth

```text
Elephants
```

#### YOLO Predictions

```text
Horse
Sheep
```

#### Observation

The model successfully localized the animals using bounding boxes but failed to classify them correctly.

Although the objects were detected, they were mapped to visually similar COCO classes rather than being identified as elephants.

---

## Failure Analysis

The wildlife misclassification results reveal a significant domain mismatch between the model's training data and the target environment.

YOLOv8n was pretrained using RGB photographs from the COCO dataset, whereas the project dataset consists of aerial thermal imagery.

The model was able to recognize the presence of large animal-shaped objects but struggled to associate thermal signatures with their correct wildlife classes.

These findings suggest that localization transfer occurs more successfully than classification transfer when moving from RGB data to thermal imagery.

---

## Challenges Encountered

Several challenges emerged during baseline implementation.

One challenge involved determining whether classification errors were caused by dataset characteristics or limitations of the pretrained model.

Another challenge involved evaluating performance without quantitative detection metrics since the dataset has not yet been converted into YOLO label format.

Interpreting thermal imagery itself also required additional effort because objects appear significantly different from conventional RGB images.

---

## Week 2 Outcome

A working baseline detection pipeline was successfully established using a COCO-pretrained YOLOv8n model.

The baseline demonstrated encouraging human detection performance but also exposed substantial limitations in wildlife classification.

These findings justify the need for model fine-tuning and provide an initial benchmark against which future improvements can be measured.

---

# Week 3: Fine-Tuning

*Pending implementation.*

### Planned Objectives

- Convert CSV annotations to YOLO format.
- Create YOLO training labels.
- Generate dataset configuration files.
- Fine-tune YOLOv8 on BIRDSAI.
- Compute Precision, Recall, F1-Score, mAP@50 and mAP@50-95.
- Perform error analysis.
---
Dataset Preparation
The BIRDSAI dataset contains thermal images organized into sequences, with corresponding CSV files containing object annotations. Since the original annotations were not directly compatible with YOLO, a preprocessing pipeline was developed to prepare the dataset for object detection.

The following activities were completed:
- Inspected the image files and annotation files to understand the dataset structure.
- Checked image readability and identified variations in image dimensions.
- Matched annotation frame numbers to their corresponding image filenames.
- Converted CSV annotations into YOLO-format label files.
- Normalized bounding-box coordinates using the actual dimensions of each image.
- Used two detection classes: animal (class 0) and human (class 1).
- Included sequence identifiers in output filenames to prevent images from different sequences from overwriting one another.
---
1. Objective
The objective of Week 3 was to prepare the BIRDSAI thermal wildlife dataset for model training, convert its annotations into YOLO-compatible labels, organize the data into training and validation sets, and initiate fine-tuning of YOLOv8n.

This stage aims to adapt the model to thermal imagery and the project's two main detection classes: animals and humans.

Dataset Validation
The dataset was checked to ensure that images and labels were correctly paired and that the labels followed the expected YOLO format.

The original training partition contained 40,661 readable images, with no unreadable images identified during the initial inspection.

The dataset was then organized into training and validation subsets using a sequence-level split. This helps prevent images from the same sequence from appearing in both subsets.
Measurement	Training set	Validation set
Images	18,028	3,181
Sequences	27	5
Bounding boxes	41,345	45,854

A total of 21,209 image-label pairs and 87,199 bounding-box annotations were generated. These included 74,668 animal boxes and 12,531 human boxes.

Structural validation confirmed that:
- Every exported image had a corresponding label file.
- Every label file had a corresponding image.
- Class IDs were within the expected range.
- Bounding-box coordinates were normalized between 0 and 1.
- Training and validation sequences did not overlap.
- No invalid bounding boxes were found during conversion.
Although the structural checks passed, visual inspection is still necessary to confirm that the bounding boxes correctly align with objects in the thermal images.
---

## Annotation Analysis

The original annotation format followed the Multiple Object Tracking (MOT) structure:

```text
frame_number
object_id
x
y
width
height
class
species
occlusion
noise
```

The project uses the broad detection classes:

```text
0 = Animal
1 = Human
```

Species information remains available for later analysis but was not used during the first fine-tuning phase.

### Why the Annotations Were Important

The annotations formed the foundation of the entire training process.

Without annotations, the model would only see thermal images without any knowledge of object locations or object categories.

The annotations provided:

- Object location through bounding box coordinates.
- Object category through class labels.
- Tracking information through object identifiers.
- Species information for future extensions.

These annotations were ultimately responsible for teaching the model the distinction between animals and humans in thermal imagery.

---

## CSV to YOLO Conversion

The original MOT annotations could not be used directly by YOLO.

A conversion pipeline was developed to transform the dataset into YOLO format.

Original annotation format:

```text
x
y
width
height
class
```

Converted YOLO format:

```text
class
x_center
y_center
width
height
```

All coordinates were normalized based on the dimensions of their corresponding image.

---

## Label Generation

Following annotation conversion:

```text
21,209 image-label pairs
```

were generated.

A total of:

```text
87,199 object annotations
```

were successfully converted into YOLO training labels.

The converted labels preserved the project detection scheme:

```text
0 = Animal
1 = Human
```

---

## Dataset Organization and Splitting

The processed dataset was reorganized into the structure expected by the Ultralytics YOLO framework.

```text
data/
└── processed/
    ├── images/
    │   ├── train/
    │   └── val/
    │
    └── labels/
        ├── train/
        └── val/
```

The dataset was split into:

```text
85% Training
15% Validation
```

The validation subset was created to monitor model performance on unseen samples during training and reduce the risk of overfitting.

The original TestReal partition remains reserved for future final evaluation.

5. Fine-Tuning Configuration
Fine-tuning was initiated using YOLOv8n initialized with pretrained weights and the prepared BIRDSAI dataset.

The initial training configuration was:
Parameter	Configuration
Model	YOLOv8n
Initial weights	yolov8n.pt
Training epochs	5
Image size	640 × 640
Batch size	2
Training device	CPU
Workers	0

A small batch size was selected because the available laptop has 8 GB of RAM and no dedicated GPU available for this training run. Consequently, training is taking considerable time.

At the time of this report, fine-tuning had started, but final training results had not yet been confirmed. Therefore, no claim of improved detection performance can be made at this stage.
---

# Week 4: Controlled Improvements

*Pending implementation.*

---

# Week 5: Tracking + Robustness

*Pending implementation.*

---

# Week 6: Final Integration

*Pending implementation.*
