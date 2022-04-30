# E-SAI
### [Official Website](https://dvs-whu.cn/projects/esai/) for Learning to See Through with Events.

Although synthetic aperture imaging (SAI) can achieve the seeing-through effect by blurring out off-focus foreground occlusions while recovering in-focus occluded scenes from multi-view images, its performance is often deteriorated by very dense occlusions and extreme lighting conditions. To address the problem, this paper presents an Event-based SAI (E-SAI) method by relying on the asynchronous events with extremely low latency and high dynamic range acquired by an event camera. 
Specifically, the collected events are first refocused by a Refocus-Net module through aligning in-focus events while scattering out off-focus ones. Following that, a hybrid network composed of spiking neural networks (SNNs) and convolutional neural networks (CNNs) is proposed to encode the spatio-temporal information from the refocused events and reconstruct a visual image of the occluded targets.

<img src="img/pipeline.png" height="200">

Exhaustive experiments demonstrate that our proposed E-SAI method can achieve remarkable performance in dealing with very dense occlusions and extreme lighting conditions and produce high-quality images from pure event data.

Previous version has been published in CVPR'21 [**Event-based Synthetic Aperture Imaging with a Hybrid Network**](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Event-Based_Synthetic_Aperture_Imaging_With_a_Hybrid_Network_CVPR_2021_paper.html), which is selected as one of the Best Paper Candidates.


## Environment setup
- Python 3.6
- Pytorch 1.6.0
- torchvision 0.7.0
- opencv-python 4.4.0
- NVIDIA GPU + CUDA
- numpy, argparse, matplotlib
- [sewar](https://github.com/andrewekhalel/sewar), [lpips](https://github.com/richzhang/PerceptualSimilarity) (for evaluation, optional)

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment with the above dependencies as follows.
<br>
Please make sure to adapt the CUDA toolkit version according to your setup when installing torch and torchvision.
```
conda create -n esai python=3.6
conda activate esai
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Download model and data
### Pretrained Model
Pretrained model can be downloaded via Baidu Net Disk. 
<br>
HybridNet and RefocusNet: [**Baidu Net Disk**](https://pan.baidu.com/s/1iqBrwwgf2bE_ztimJhWjmA) (Password: u8a4)
<br>
Note that the network structure is slightly different from the model in our CVPR paper.

### Example Data
Some example data is available. The whole dataset will be released soon. 
<br>
Example Data: [**Baidu Net Disk**](https://pan.baidu.com/s/1AC0KjsMdWNznXzwhE4MVdg) (Password: dklm) or [**Google Drive**](https://drive.google.com/drive/folders/1kHBANtcQDi7GyBWyykvgFKjTGH36V1-O?usp=sharing).
<br>
You can also check out our [**EF-SAI**](https://github.com/smjsc/EF-SAI) dataset via [**One Drive**](https://onedrive.live.com/?authkey=%21AMvAPOnuudsYx1I&id=7ABD0A750B262518%214850&cid=7ABD0A750B262518) or [**Baidu Net Disk**](https://pan.baidu.com/s/1VKbt0hoh44Ax7QX4sblBKQ?pwd=3tgv#list/path=%2F).


## Quick start
### Initialization
Change the parent directory to './codes/'
```
cd codes
```
- Create directories
```
mkdir -p PreTraining Results Example_data/{Raw,Processed}
```
- Copy the pretrained model to directory './PreTraining/'
- Copy the event data and the corresponding occlusion-free APS images to directories './Example_data/Raw/Event/' and  './Example_data/Raw/APS/'

### E-SAI+Hybrid (M)
Run E-SAI+Hybrid with manual refocusing module.
- Preprocess event data with manual refocusing
```
python Preprocess.py --do_event_refocus=1 --input_event_path=./Example_data/Raw/Event/ --input_aps_path=./Example_data/Raw/APS/
```
- Run reconstruction (using only HybridNet)
```
python Test_ManualRefocus.py --reconNet=./PreTraining/Hybrid.pth --input_path=./Example_data/Processed/Event/ --save_path="./Results/Test/"
```
The reconstruction results will be saved at save_path (default: './Results/Test/').

### E-SAI+Hybrid (A)
Run E-SAI+Hybrid with auto refocusing module.
- Preprocess event data without refocusing
```
python Preprocess.py --do_event_refocus=0 --input_event_path=./Example_data/Raw/Event/ --input_aps_path=./Example_data/Raw/APS/
```
- Run reconstruction (using HybridNet and RefocusNet)
```
python Test_AutoRefocus.py --reconNet=./PreTraining/Hybrid.pth --refocusNet=./PreTraining/RefocusNet.pth --input_path=./Example_data/Processed/Event/ --save_path="./Results/Test/"
```
The reconstruction results will be saved at save_path (default: './Results/Test/'). 
<br>
This code will also calculate the Max Pixel Shift Error (MPSE) and save the result in './Results/Test/MPSE.txt'.

### Evaluation
Evaluate the reconstruction results with metrics PSNR, SSIM and LPIPS.
- Copy the occlusion-free APS images in './Example_data/Raw/APS/' to directory './Results/True/'
- Run evaluation
```
python Evaluation.py
```
This code will create an IQA.txt file containing the quantitative results in './Results/IQA.txt'.


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
