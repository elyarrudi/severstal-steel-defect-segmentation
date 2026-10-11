# Severstal Steel Defect Segmentation

Pixel-level segmentation of steel surface defects with a U-Net implemented in PyTorch. The network is trained from random initialization: the notebook does not load a pretrained backbone or pretrained segmentation weights.

The training code is `industrial_segment.ipynb`. It was written for Google Colab and reads the dataset from Google Drive. This repository contains the notebook, the selected checkpoint `best_model.pth`, and the saved validation and test visualizations.

## Result

![Validation examples for all four defect classes](predictions/results/portfolio_visualizations/all_classes_portfolio.png)

Selected checkpoint, epoch 66 of `predictions/results/training_history.csv`, on the validation split:

| Metric | Score |
| --- | ---: |
| Mean defect Dice | 0.60492 |
| Mean defect IoU | 0.45045 |
| Validation loss | 0.24755 |

| Class | Dice | IoU | Annotation rows |
| ---: | ---: | ---: | ---: |
| 1 | 0.50116 | 0.33437 | 897 |
| 2 | 0.41440 | 0.26136 | 247 |
| 3 | 0.73277 | 0.57824 | 5,150 |
| 4 | 0.77136 | 0.62782 | 801 |

Class 2 is the weakest class and also the smallest class in the annotation table. These scores are validation scores. The Severstal test images have no ground-truth masks in this pipeline, so the files under `predictions/test_predictions/` are qualitative only.

An earlier checkpoint, epoch 39, is stored in `predictions/results/final_metrics_improved.csv` (mean defect Dice 0.60231). `final_metrics_balanced.csv` matches epoch 66.

Per-class comparisons are in `predictions/results/portfolio_visualizations/`. The training plot is `predictions/results/balanced_training_results.png`.

## Dataset

[Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data) on Kaggle. The notebook expects `severstal-steel-defect-detection.zip` on Google Drive.

The annotation table used for training has 7,095 rows and 6,666 unique defective images. Masks are run-length encoded. The notebook decodes them at 256×1600, then resizes images and masks to 256×512. The model predicts five labels: background plus defect classes 1–4. The CSV identifies classes by id only.

The split is by image id, 80/20, seed 42: 5,332 training images and 1,334 validation images.

`train.csv`, `train_images/`, and `test_images/` are listed in `.gitignore` and are not part of this repository.

## What the notebook trains

- U-Net in PyTorch: double convolutions, max pooling, transposed convolutions, skip connections, and a 1×1 output layer. Base width is 32 channels.
- Training augmentation is horizontal flip (p=0.5) and random brightness/contrast. Validation is not augmented.
- Class weights use `sqrt(max_frequency / class_frequency)`, with background weight 0.25.
- The first training section optimizes `0.5 * weighted cross-entropy + 0.5 * Dice`. The later sections use `0.4 * weighted cross-entropy + 0.6 * Dice`. Dice ignores the background class.
- Optimizer is AdamW. The first section starts at learning rate `1e-3`. Mixed precision is used when CUDA is available. Seed is 42.
- `training_history.csv` records 66 epochs. The highest mean defect Dice in that file is epoch 66.

## Repository layout

```text
industrial/
├── industrial_segment.ipynb
├── best_model.pth
├── predictions/
│   ├── results/
│   │   ├── training_history.csv
│   │   ├── final_metrics_balanced.csv      # epoch 66
│   │   ├── final_metrics_improved.csv      # epoch 39
│   │   ├── balanced_training_results.png
│   │   └── portfolio_visualizations/
│   └── test_predictions/
│       ├── colored_masks/                  # 2,722 images
│       ├── overlays1/                      # 888 images
│       └── overlays/                       # 2,720 images, gitignored
└── README.md
```

`predictions/test_predictions/overlays/` is gitignored because of its size. `overlays1/` is the smaller sample. `last_checkpoint.pth` is also gitignored.

## Run the notebook

The notebook calls `drive.mount("/content/drive")` and looks for the dataset zip at `/content/drive/MyDrive/severstal-steel-defect-detection.zip`. Checkpoints are written to `/content/drive/MyDrive/Severstal_UNet_Training`.

To reproduce training, open the notebook in Colab, put the Kaggle zip at that Drive path, and run the cells. A local rerun needs those paths changed. The dataset and the Colab environment are not included here.

## Limits

- Reported Dice and IoU are on the validation split, not on a held-out test set with labels.
- Class 2 remains well below classes 3 and 4.
- Images are resized from 256×1600 to 256×512 before training.
- The notebook is a Colab training record. It does not include a Docker inference app.
