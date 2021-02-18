# endscope_diagnosis

## 学習

### 1，10分割したテキストを次のように配置する．
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
3N100_{num}.txtの中身は次のようなフォーマットとなっている
```
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
{image_path} {label(hyperplasticpolyp=0,Adenoma=1,Normalmucosa=2)}
︙

```
example:
```
../../Images/cropped/A16/A16-00097.jpg 0
../../Images/cropped/A16/A16-00098.jpg 0
../../Images/cropped/A16/A16-00099.jpg 0
```
### 2，学習を行う
monitor.pyを用いて学習を行う.
exptmgr.pyを行うことで一括して学習を行うことができる．

example:
```
cd Resnet_tf2/sorce
python monitor.py ../text/3n100/3n100_01.txt ./config.ini
```
```
cd Resnet_tf2/sorce
python exptmgr.py ../text/3n100 ./config.ini -c
```

## 検証
inspector.pyを用いて識別が不正解だった画像のラベルとモデルの出力結果をリスト化できる.
```
cd Resnet_tf2/sorce
python inspector.py ../products/product_3N100/wh5/3N100_01_w.h5 ../text/3n100/3n100_01.txt
```
gradcam.pyはGradCAMを行うためのスクリプトである．
```
cd Resnet_tf2/sorce
python gradcam.py ../products/product_3N100/json/3N100_01.json ../products/product_3N100/wh5/3N100_01_w.h5 ../../Images/cropped/A01/
```

igen.py,iread.py,iseq.pyは画像の入力に使用している．

##使用した環境について
- Ubuntu 18.04.5 LTS
- CUDA   10.0
- CUDNN  7.6.5.32 
- Python 3.6.9
- numpy  1.20.0.dev0+8b15e57
- opencv-python 4.1.0.25
- tensorflow-gpu 2.0.0
