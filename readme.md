# CIFAR-10 Classify

This project uses the ResNet-18 and customResNet models to train on the CIFAR-10 dataset and conduct image classification tests.

## Create an environment

```
# 创建环境
conda env create -f environment.yml

# 或者从当前目录创建
conda env create -n pic_classify -f environment.yml
```

## Training

Run "train.py" to train the modified ResNet-10 model that is adapted to the size of CIFAR-10 images.

```
python train.py
```

Run the custom_train.py script to train the customResNet model that has been built.

```
python custom_train.py
```

## Testing

Remove the comment selection for the test model in test.py

## Drawing a chart

Run "draw_chart.py" to generate the curves of loss rate and accuracy change during the training process based on the log file.

```
python draw_chart.py
```







