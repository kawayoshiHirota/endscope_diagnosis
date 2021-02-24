# endscope_diagnosis

## learning

### 1. 10 text files should be located as follows. Each text file corresponds to each sample of learning data.
```
text
└── 3N100
    ├── 3N100_01.txt
    ├── 3N100_02.txt
    ├── 3N100_03.txt
    ├── 3N100_04.txt
    ├── 3N100_05.txt
    ├── 3N100_06.txt
    ├── 3N100_07.txt
    ├── 3N100_08.txt
    ├── 3N100_09.txt    
    └── 3N100_10.txt
```
The format of 3N100_{num}.txt is defined as follows:
```
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
:
:
```
example:
```
../../Images/cropped/A16/A16-00097.jpg 0
../../Images/cropped/A16/A16-00098.jpg 0
../../Images/cropped/A16/A16-00099.jpg 0
```
### 2. learning
You can use monitor.py for training neural networks.
exptmgr.py is a script for the 10-fold cross-validation, namely, a single run of exptmgr.py can generate 10 models in turn.

example:
```
cd ResNet_tf2/sorce
python monitor.py ../text/3N100/3N100_01.txt ./config.ini
```
```
cd ResNet_tf2/sorce
python exptmgr.py ../text/3N100 ./config.ini -c
```

## validation
inspector.py yields the list of the label of teacher value and corresponding output by using the trained model for misclassified test data.
```
cd ResNet_tf2/sorce
python inspector.py ../products/product_3N100/wh5/3N100_01_w.h5 ../text/3N100/3N100_01.txt
```
gradcam.py is for GradCAM.
```
cd ResNet_tf2/sorce
python gradcam.py ../products/product_3N100/json/3N100_01.json ../products/product_3N100/wh5/3N100_01_w.h5 ../../Images/cropped/A01/
```

igen.py, iread.py, and iseq.py are used for preprocessing of input images.

## compatibility
- Ubuntu 18.04.5 LTS
- CUDA   10.0
- CUDNN  7.6.5.32 
- Python 3.6.9
- numpy  1.20.0.dev0+8b15e57
- opencv-python 4.1.0.25
- tensorflow-gpu 2.0.0
