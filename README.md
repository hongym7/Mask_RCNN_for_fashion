# Kaggle Competition
***  

## Description
In this competition, we challenge you to develop algorithms that will help with an important step towards automatic product detection – to accurately assign segmentations and attribute labels for fashion images.

## Data
* train imageset
* test imageset
* label_description_json
* sample_submission.csv
* train.csv


## Result
1.  

    LR = 1e-5
    EPOCHS = [40, 100, 200]
    augmentation = iaa.Sequential([
      iaa.Fliplr(0.5),
      iaa.Crop(percent=(0.08, 0.15))
    ])
2.

    LR = 1e-5
    EPOCHS = [50, 100, 200]
    augmentation = iaa.Sequential([
      iaa.Fliplr(0.5),
      iaa.Cutout(nb_iterations=(1, 4), size=0.2, squared=False),
      iaa.Crop(percent=(0.08, 0.15)),
      iaa.PerspectiveTransform(scale=(0.01, 0.15))
    ])
    
3. Best Result

    LR = 1e-3
    EPOCHS = [50, 150, 400, 700]
    augmentation = iaa.Sequential([
      iaa.Fliplr(0.5),
      iaa.Crop(percent=(0.08, 0.15))
    ])

