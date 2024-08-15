# AOI detection system
# structure
```
AOI
├── dataset
│   ├── origin
│   └── sample
├── label
├── weights
└── segmentation
```
---
## detect
```terminal
python main.py --op det 
```
---
## label -*al* : auxiliary tool for sample label
Aruguments :
```
--text :  name of label text,being created if not exist
--oip  :  the folder of origin data
--sip  :  the folder of segmentation of origin data
--srcdes : saving destination of sample
```
Cmd :
```terminal
python main.py --oip dataset\origin --sip dataset\segmentation --text label\label.txt --srcdes dataset\train --op al
```
---
## segmentation -*sg* : Segmentation for CRS 
Aruguments :
```
--sip : path of folder that store results of segmentation.
--oip : the data folder that will be transformed to 
segmentation image.
```
```terminal
python main.py --op sg --sip segmentation --oip datset\origin
```
---
# training
```
--label : text, default = label\label.txt
--dataset : folder default = dataset\train
--batch_size : default 32
--test_size : 0.1
--epochs : default 100
--pthname : 
```
```terminal
python train 
```
