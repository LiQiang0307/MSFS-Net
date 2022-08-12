# MSFS-Net
Multi-scale frequency separation network for image deblurring


## Installation
```
Python 3.7.13
pytorch  1.9.0
CUDA 10.2
scikit-image
opencv-python
Tensorboard
```


## Pretrained Models

We provide our pre-trained models. You can test our network following the instruction below.

链接：https://pan.baidu.com/s/1FwHEuyivhCP_BynZC0Ayjw 
提取码：0516

| weights    | training dataset |
| --------   | -----:  |
| model.pkl     | GoPro  | 
| model_R.pkl        |   RealBlur-R   | 
| model_J.pkl      |   RealBlur-J   | 

## Dataset

prepare datasets

#### GoPro
* Download deblur dataset from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html).
* Unzip files `dataset` folder.
* Preprocess dataset by running the command below:

  `python data/preprocessing.py`

* After preparing data set, the data folder should be like the format below:
  ```
  GOPRO
  ├─ train
  │ ├─ blur    % 2103 image pairs
  │ │ ├─ xxxx.png
  │ │ ├─ ......
  │ │
  │ ├─ sharp
  │ │ ├─ xxxx.png
  │ │ ├─ ......
  │
  ├─ test    % 1111 image pairs
  │ ├─ ...... (same as train)

  ```
#### HIDE
* Download deblur dataset from the [HIDE dataset](https://github.com/joanshen0508/HA_deblur)
* Preprocess dataset by running the command below:

  `python data/HIDE.py`
  
  **note**: Please change the path of the dataset location in the code
  
  **format**：the same as GoPro datasets

#### RealBlur
* Download deblur dataset from the [RealBlur dataset](https://github.com/rimchang/RealBlur)
* The data folder should be like the format as same as GoPro datasets.


## Test
#### GoPro and HIDE
To test MSFS-Net，run the command below:
```
python main.py --model_name "MSFS-Net" --mode "test" --data_dir "dataset/GOPRO" --test_model "model.pkl"
```
**note**:You should change line 32 of `main.py` to `model=build_net()` 

or to test MSFS-Net-Local, run the command below:
```
python main.py --model_name "MSFS-Net-Local" --mode "test" --data_dir "dataset/GOPRO" --test_model "model.pkl"
```

**note**:You should change line 32 of `main.py` to `model=build_arch_net()` 
#### RealBlur
  The run command is the same as above，but you should change line 15 of `main.py` to `from eval_R import _eval` 

## PSNR and SSIM
We measured PSNR and SSIM using matlab functions.
