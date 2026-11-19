# BIRDSAI simulated human training sample

The source is the official [BIRDSAI TrainSimulation release](https://lila.science/datasets/conservationdrones), distributed under CDLA-Permissive 1.0. The existing local BIRDSAI export came from TrainReal and TestReal. TrainSimulation adds synthetic aerial thermal scenes, with published bounding boxes and sequence identifiers.

The source inventory contained 124 annotation CSVs, including 68 sequences with at least one human annotation. The sample takes five interior quantiles of the human-positive frame IDs per sequence, for sequences with at least five such frames. This spreads the selection across each sequence and avoids picking five consecutive frames. Only thermal JPEGs are selected. Each exported YOLO label preserves all source boxes for that frame, including animal boxes. The image and annotation source URLs, sequence IDs, frame IDs and image hashes are recorded in the sample manifest.

Sample location: `data/external/birdsai_sim_human_sample/`. It contains 335 verified image-label pairs across 67 sequences, with 439 human boxes and 842 animal boxes. During conversion, 49 boxes were clipped to image boundaries and zero were discarded. The images are kept out of Git along with other dataset files.

The median native human box heights were 28 pixels in real training, 9 pixels in real validation, and 7 pixels in the synthetic sample. Of the 439 synthetic human boxes, 313 were below 16 pixels tall. This sample supplements the small end of the human box size distribution.

Use this sample as supplemental training data in a controlled comparison with a real-only run from the same checkpoint. Keep complete recordings together, and keep the existing real validation and TestReal images separate from training. Synthetic examples can add training variety. Performance on actual thermal cameras still needs to be measured with real labeled recordings.

Citation: Bondi et al., [BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos](https://openaccess.thecvf.com/content_WACV_2020/papers/Bondi_BIRDSAI_A_Dataset_for_Detection_and_Tracking_in_Aerial_Thermal_WACV_2020_paper.pdf), WACV 2020.
