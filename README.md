# Video Semantic Segmentation with Spatial-Temporal Fusion and Memory-Augmented Refinement
This repository is the official implementation of "Video Semantic Segmentation with Spatial-Temporal Fusion and Memory-Augmented Refinement" (submitted to IEEE Transactions on Circuits and Systems for Video Technology(TCSVT) 2021). It is designed for accurate video semantic segmentation task. It is based on mmsegmentaion.

## Install & Requirements
The code has been tested on pytorch=1.8.1 and python3.7. Please refer to `requirements.txt` for detailed information.

**To Install python packages**
```
pip install -r requirements.txt
```
## Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets.

Your directory tree should be look like this:
````bash
$DAVSS_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
├── camvid
│   ├── label
│   │   ├── segmentation annotations
│   └── video_image
│       ├── 0001TP
│           ├── decoded images from video clips
│       ├── 0006R0
│       └── 0016E5
│       └── Seq05VD
````
## Train and test
For example, train our proposed method on Cityscapes on 4 GPUs:
````bash
# train STF
cd ST_Memory/exp/cityscapes/psp50/STF/script
bash train.sh
# generate key-value memory bank
cd ST_Memory/exp/cityscapes/psp50/generate_memory/script
bash test.sh
# train MAR
cd ST_Memory/exp/cityscapes/psp50/MAR/script
bash train.sh
````

For example, test our proposed method on Cityscapes validation set:
````bash
cd ST_Memory/exp/cityscapes/psp50/MAR/script
bash test.sh
````

## Trained model
We provide trained model on Cityscapes datasets. Please download models from:
| model | Link |
| :--: | :--: |
| psp50+STF+MAR | [BaiduYun(Access Code:xq7x)](https://pan.baidu.com/s/1uFVitsS47oq58Z3RAiCQrQ)
