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
from utils import crop, refocus, calculate_MPSE
from Networks.Hybrid import HybridNet
from Networks.Refocus import RefocusNet
from Event_Dataset import TestSet_AutoRefocus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(opt):
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    
    ## load refocusNet
    refocusNet = RefocusNet()
    refocusNet = torch.nn.DataParallel(refocusNet)
    refocusNet.load_state_dict(torch.load(opt.refocusNet, map_location={'cuda:0':'cpu'}))
    refocusNet.to(device)
    refocusNet = refocusNet.eval()
    
    ## load reconNet, use HybridNet here
    reconNet = HybridNet()
    reconNet = torch.nn.DataParallel(reconNet)
    reconNet.load_state_dict(torch.load(opt.reconNet, map_location={'cuda:0':'cpu'}))
    reconNet.to(device)
    reconNet = reconNet.eval()
    
    ## prepare dataset
    testDataset = TestSet_AutoRefocus(opt.input_path)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)
        
    f = open(opt.save_path + 'MPSE.txt',"a+") # open a txt to save MPSE results
    MPSEs = []
    with torch.no_grad():
        for i, (eData, diff_t, depth, fx) in enumerate(testLoader):
            print('Processing data %d ...'%i)
            eData = eData.to(device)
            
            # apply refocus net
            predictInp = eData
            psi = refocusNet(predictInp)
            
            # apply recon net
            reconInp = refocus(eData, psi, diff_t)
            reconInp = crop(reconInp) # crop roi
            timeWin = reconInp.shape[1]
            outputs = reconNet(reconInp, timeWin)
            
            # calculate MPSE and save results
            MPSE = calculate_MPSE(psi.cpu(), diff_t, depth, fx=fx)
            MPSEs.append(MPSE)
            f.write('MPSE of data %d:  %.5f \n'%(i,MPSE))
            name = opt.save_path + '%04d' % i + '.png'
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
    parser.add_argument("--input_path", type=str, default="./Example_data/Processed/Event/", help="input data path") 
    parser.add_argument("--save_path", type=str, default="./Results/Test/", help="saving path")

    opt = parser.parse_args()
    test(opt)
    
