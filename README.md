# Seeing via Contexts and Experiences: Dual Memory-Guided Refinements for Video Semantic Segmentation
This repository is the official implementation of "Seeing via Contexts and Experiences: Dual Memory-Guided Refinements for Video Semantic Segmentation".

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
# train CRM
cd DMR/exp/cityscapes/psp50/CRM/script
bash train.sh
# generate experience-based memory bank
cd DMR/exp/cityscapes/psp50/generate_memory/script
bash test.sh
# train ERM
cd DMR/exp/cityscapes/psp50/ERM/script
bash train.sh
````

For example, test our proposed method on Cityscapes validation set:
````bash
cd DMR/exp/cityscapes/psp50/ERM/script
bash test.sh
````

## Trained model
We provide trained model on Cityscapes datasets. Please download models from:
| model | Link |
| :--: | :--: |
| psp50+CRM+ERM | [BaiduYun(Access Code:xq7x)](https://pan.baidu.com/s/1uFVitsS47oq58Z3RAiCQrQ)
