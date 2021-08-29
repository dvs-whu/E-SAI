from __future__ import print_function
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']="0" # choose GPU
import torch
import cv2
import numpy as np
from torch.nn import functional as F
from Networks.Hybrid import HybridNet
from Networks.Refocus import RefocusNet
from Event_Dataset import TestSet_AutoRefocus
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

def refocus(data, psi, diff_t):
    ## refocus events based on timestep diff_t(b,30), psi(b,1)
    diff_t = diff_t.to(psi.device)
    refocused_data = torch.zeros_like(data).to(data.device)
    for i in range(data.shape[0]): # batch
        current_diff_t = diff_t[i,:] #(30)
        current_psi = psi[i,:] #(2)
        theta = torch.zeros((data.shape[1], 2, 3), dtype = torch.float).to(psi.device) #(step,2,3)
        theta[:,0,0] = theta[:,1,1] = 1 # no zoom in/out
        theta[:,0,2] = current_psi[0] * current_diff_t # x_shift
        theta[:,1,2] = current_psi[1] * current_diff_t # y_shift
        grid = F.affine_grid(theta, data[i,:].squeeze().size())
        refocused_data[i,:] = F.grid_sample(data[i,:].squeeze(), grid)
    
    return refocused_data

def calculate_MPSE(psi, diff_t, depth, width=346, v=0.177, fx=320.132621):
    # calculate horizontal MPSE
    psi_x = psi.squeeze()[0]
    pred_depth = (2*fx*v) / (-1*psi_x*width)
    
    max_abs_diff_t = torch.max(torch.abs(diff_t))
    max_pix_shift_real = max_abs_diff_t * fx * v / depth 
    max_pix_shift_pred = max_abs_diff_t * fx * v / pred_depth
    MPSE = np.abs(max_pix_shift_real - max_pix_shift_pred)
    return MPSE

def test(opt):
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)
    
    ## load refocusNet
    refocusNet = RefocusNet()
    refocusNet = torch.nn.DataParallel(refocusNet)
    refocusNet.load_state_dict(torch.load(opt.refocusNet, map_location={'cuda:0':'cpu'}))
    refocusNet.to(device)
    refocusNet = refocusNet.eval()
    
    ## load reconNet, use Hybrid here
    reconNet = HybridNet()
    reconNet = torch.nn.DataParallel(reconNet)
    reconNet.load_state_dict(torch.load(opt.reconNet, map_location={'cuda:0':'cpu'}))
    reconNet.to(device)
    reconNet = reconNet.eval()
    
    ## prepare dataset
    testDataset = TestSet_AutoRefocus(opt.input)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)
        
    f = open(opt.save + 'MPSE.txt',"a+") # open a txt to save MPSE results
    MPSEs = []
    with torch.no_grad():
        for i, (eData, diff_t, depth, fx) in enumerate(testLoader):
            print('Processing data %d ...'%i)
            eData = eData.to(device)
            
            # refocus net
            predictInp = eData
            psi = refocusNet(predictInp)
            
            # recon net
            reconInp = refocus(eData, psi, diff_t)
            reconInp = crop(reconInp) # crop roi
            timeWin = reconInp.shape[1]
            outputs = reconNet(reconInp, timeWin)
            
            # calculate MPSE and save results
            MPSE = calculate_MPSE(psi.cpu(), diff_t, depth, fx=fx)
            MPSEs.append(MPSE)
            f.write('MPSE of data %d:  %.5f \n'%(i,MPSE))
            name = opt.save + '%04d' % i + '.png'
            img = (outputs[0,:].permute(1,2,0)*255).cpu()
            img = img.numpy()
            cv2.imwrite(name, img)
    
    f.write('Mean MPSE = %.5f '%(np.array(MPSEs).mean()))
    f.close()
    print('Completed !')
    

if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Test E-SAI+Hybrid (A)")
    parser.add_argument("--refocusNet", type=str, default="./PreTraining/RefocusNet.pth", help="refocusNet path")
    parser.add_argument("--reconNet", type=str, default="./PreTraining/Hybrid.pth", help="reconNet path")
    parser.add_argument("--input", type=str, default="./Example_data/Processed/Event/", help="input data path") 
    parser.add_argument("--save", type=str, default="./Results/Test/", help="saving path")

    opt = parser.parse_args()
    test(opt)
    
