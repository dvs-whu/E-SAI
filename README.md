# E-SAI
### [Official Website](https://dvs-whu.cn/projects/esai/) for Event-based Synthetic Aperture Imaging.

Our CVPR paper can be found here:
<br>
###[**Event-based Synthetic Aperture Imaging with a Hybrid Network**](https://arxiv.org/abs/2103.02376)
<br>
Xiang Zhang*, Wei Liao*, Lei Yu&dagger;, Wen Yang, Gui-Song Xia 
<br>
Wuhan University
<br>
(* Equal Contribution &dagger; Corresponding Author)

<img src="img/pipeline.png" height="200">

## Environment setup
- Python 3.6
- Pytorch 1.6.0
- torchvision 0.7.0
- python-opencv 4.4.0
- NVIDIA GPU + CUDA
- numpy, argparse
- [sewar](https://github.com/andrewekhalel/sewar), [lpips](https://github.com/richzhang/PerceptualSimilarity) (for evaluation, optional)

## Download model and data
### Pretrained Model
Pretrained model can be downloaded via Baidu Net Disk. 
<br>
[**HybridNet and RefocusNet**](https://pan.baidu.com/s/1_4aIuVq1TwMgF79F8XMZ2Q) (Password: urbg)
<br>
Note that the network structure is slightly different from the model in CVPR paper.

### Data
Some example data is available now. The whole dataset will be released soon.
<br>
[**ExampleData**](https://pan.baidu.com/s/1O4KVdsV3pvqnIQNj9hQQkg) (Password: m2uv)
<br>

## Quick start
### Initialize
```
cd codes
```
- Create directories
```
mkdir -p PreTraining Results Example_data/{Raw,Processed}
```
- Copy the pretrained model to directory './PreTraining/'
- Copy the data to directory './Example_data/Raw/'
### Run E-SAI+Hybrid (M)
- Preprocessing event data with manual refocusing
```
python Preprocess.py --do_event_refocus=1 --input_event_path=./Example_data/Raw/Event/ --input_aps_path=./Example_data/Raw/APS/
```
- Reconstruction (only apply HybridNet)
```
python Test_ManualRefocus.py --model=./PreTraining/Hybrid.pth --input_path=./Example_data/Processed/Event/ --save_path="./Results/Test/"
```
### Run E-SAI+Hybrid (A)
- Preprocessing event data without refocusing
```
python Preprocess.py --do_event_refocus=0 --input_event_path=./Example_data/Raw/Event/ --input_aps_path=./Example_data/Raw/APS/
```
- Reconstruction (apply HybridNet and RefocusNet)
```
python Test_ManualRefocus.py --reconNet=./PreTraining/Hybrid.pth --refocusNet=./PreTraining/RefocusNet.pth --input_path=./Example_data/Processed/Event/ --save_path="./Results/Test/"
```
### Evaluation
- Copy the ground truth images to directory './Results/True/'
- Run evaluation
```
python Evaluation.py
```
This code will create a IQA.txt file containing the quantitative results in './Results/IQA.txt'.


## Citation

If you find our work useful in your research, please cite:

```
@inproceedings{zhang2021event,
  title={Event-based Synthetic Aperture Imaging with a Hybrid Network},
  author={Zhang, Xiang and Liao, Wei and Yu, Lei and Yang, Wen and Xia, Gui-Song},
  year={2021},
  booktitle={CVPR},
}
```
