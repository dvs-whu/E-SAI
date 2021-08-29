from __future__ import print_function
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']="0" # choose GPU
import torch
import cv2
from Networks.Hybrid import HybridNet
from Event_Dataset import TestSet_Hybrid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop(data, roiTL=(2,45), size=(256,256)):
    # function to crop the region of interest
    Xrange = (roiTL[1], roiTL[1] + size[1])
    Yrange = (roiTL[0], roiTL[0] + size[0])
    if data.ndim == 3:
        out = data[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    elif data.ndim == 4:
        out = data[:,:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    else:
        out = data[:,:,:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    return out

def test(opt):
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    
    ## load model and data
    Net = HybridNet()
    testDataset = TestSet_Hybrid(opt.input_path)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)
    Net = torch.nn.DataParallel(Net)
    Net.load_state_dict(torch.load(opt.model,map_location=torch.device('cpu')))
    Net.to(device)
    Net = Net.eval()
    
    with torch.no_grad():
        for i, (eData) in enumerate(testLoader):
            print('Processing data %d ...'%i)
            time_win = eData.shape[1]
            eData = eData.to(device)
            eData = crop(eData)
            outputs = Net(eData, time_win)
            
            ## saving results
            name = opt.save_path + '%04d' % i + '.png'
            img = (outputs[0,:].permute(1,2,0)*255.0).cpu()
            img = img.numpy()
            cv2.imwrite(name, img)
            
    print('Completed !')

if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Test E-SAI+Hybrid (M)")
    parser.add_argument("--model", type=str, default="./PreTraining/Hybrid.pth", help="model path")
    parser.add_argument("--input_path", type=str, default="./Example_data/Processed/Event/", help="input data path")
    parser.add_argument("--save_path", type=str, default="./Results/Test/", help="saving path")

    opt = parser.parse_args()
    
    ## test
    test(opt)
