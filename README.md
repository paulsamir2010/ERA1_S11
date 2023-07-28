# ERA1_S11
# ERA1 Assignment # 11  - TSAI ("The School of AI") - Train ResNet 18 Model and use GradCAM

Dataset = CIFAR10 Used 
Framework = PyTorch
Model = ResNet18
Epochs = 20

## Requirement and Objective
Requirement is to use Train ResNet18 on Cifar10 for 20 Epochs and use GradCAM to study the misclassified image
Use Albumentation for data Augmentation


## Organization of files in this repository

- Model is in mymodels.py

- Train and test code are in mytrain.py and mytest.py

- GradCAM code is in gradcam.py

- utility.py contains functions to fetch misclasified images and plotting misclassified images

- main.ipynb is the Main ipynb file

## MisClassified Images

![image](https://github.com/paulsamir2010/ERA1_S11/blob/main/MisClassified.jpg)

## GradCAM Visualization of MisClassified Images

![image](https://github.com/paulsamir2010/ERA1_S11/blob/main/GradCAM1.jpg)


![image](https://github.com/paulsamir2010/ERA1_S11/blob/main/GradCAM2.jpg)


Test Accuracy = 77 %
