# Data Science Bowl 2018
Repository for our attempt at Kaggle's 2018 Data Science Bowl Competition.

Currently implementing MaskRCNN based off RetinaNet.

## Training
```
python keras_maskrcnn/bin/train.py csv [annotations file] [class map]
```

## Testing
```
python test.py [test dir] [out dir] [model path]
```
This script will test the model given by the third argument on the images found in the first. This script assumes the test directory is laid out in the same manner as the competition. Easily fixed in the generator class at the top.
