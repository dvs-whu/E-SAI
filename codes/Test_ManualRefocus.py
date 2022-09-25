from __future__ import print_function
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']="0" # choose GPU
import torch
import cv2
from utils import crop, mkdir
from Networks.Hybrid import HybridNet
from Event_Dataset import TestSet_ManualRefocus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(opt):
    mkdir(os.path.join(opt.save_path, 'Test'))
    mkdir(os.path.join(opt.save_path, 'True'))
    
    ## load model and data
    Net = HybridNet()
    testDataset = TestSet_ManualRefocus(opt.input_path)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)
    Net = torch.nn.DataParallel(Net)
    Net.load_state_dict(torch.load(opt.reconNet,map_location=torch.device('cpu')))
    Net.to(device)
    Net = Net.eval()
    
    with torch.no_grad():
        for i, (eData, occ_free_aps) in enumerate(testLoader):
            print('Processing data %d ...'%i)
            time_win = eData.shape[1]
            eData = eData.to(device)
            eData = crop(eData, roiTL=(2,45), size=(256,256))
            outputs = Net(eData, time_win)
            
            ## saving results
            name = os.path.join(opt.save_path, 'Test', '%04d' % i + '.png')
            img = (outputs[0,:].permute(1,2,0)*255).cpu().numpy()
            cv2.imwrite(name, img)
            
            name = os.path.join(opt.save_path, 'True', '%04d' % i + '.png')
            occ_free_aps = crop(occ_free_aps, roiTL=(2,45), size=(256,256)).squeeze().numpy()
            cv2.imwrite(name, occ_free_aps)
            
    print('Completed !')

if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Test E-SAI+Hybrid (M)")
    parser.add_argument("--reconNet", type=str, default="./PreTraining/Hybrid.pth", help="reconNet path")
    parser.add_argument("--input_path", type=str, default="./Example_data/Processed/", help="input data path")
    parser.add_argument("--save_path", type=str, default="./Results/", help="saving path")

    opt = parser.parse_args()
    
    ## test
    test(opt)
