import numpy as np
import sys # avoid the path of ROS
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import argparse
import torch
from Event_Dataset import get_file_name
import lpips
from sewar.full_ref import psnr
from sewar.full_ref import ssim

if __name__ == '__main__':
    """
    This code will generate a txt file containing the quantitative results
    Note that the result path should have the following directory structure:
    |---Your_result_path (default: Results)
        |---Test (containing reconstruction results)
        |---True (containing ground truth)
    """
    ## parameters
    parser = argparse.ArgumentParser(description="IQA including PSNR , SSIM and LPIPS")
    parser.add_argument("--path", type=list, default=["./Results/"],  help="basic path for results")
                               
    opt = parser.parse_args()
    
    for i in range(len(opt.path)):
        print("Processing "+opt.path[i]+" ...")
        reconPath = opt.path[i] + 'Test/'
        gtPath = opt.path[i] + 'True/'
        resPath = opt.path[i] + 'IQA.txt'
    
        reconName = get_file_name(reconPath,'.png') 
        gtName = get_file_name(gtPath,'.png') 
    
        assert len(reconName) == len(gtName)
    
        psnrs = []
        ssims = []
        lpipss = []
        loss_fn_alex = lpips.LPIPS(net='alex')
        f = open(resPath,"a+")
        
        for i in range(len(reconName)):
            print("Processing %d img ... "%(i))
            # Reading data...
            nameTest = reconPath + reconName[i]
            imgTest = cv2.imread(nameTest)
            
            nameTrue = gtPath + gtName[i]
            imgTrue = cv2.imread(nameTrue)[2:2+256,45:45+256,:]
            
            # Calculating results...
            resPSNR = psnr(imgTrue,imgTest)
            resSSIM = ssim(imgTrue,imgTest,16)[0]
            w,h,c = imgTest.shape
            imgTorchTrue = torch.FloatTensor(imgTrue.reshape(c,w,h)/255).unsqueeze(0)
            imgTorchTest = torch.FloatTensor(imgTest.reshape(c,w,h)/255).unsqueeze(0)
            resLPIPS = loss_fn_alex(imgTorchTrue, imgTorchTest).squeeze().detach().numpy()
            
            # Writing data ...
            title = 'recon:'+ reconName[i] + '  v.s.  ' + 'gt:' + gtName[i] + '-------------- \n'
            f.write(title)
            res = 'PSNR: %.4f  ;SSIM: %.4f  ;LPIPS: %.4f  . \n' %(resPSNR,resSSIM,resLPIPS)
            f.write(res)
            
            psnrs.append(resPSNR)
            ssims.append(resSSIM)
            lpipss.append(resLPIPS)
        
        # Calculating mean value...
        meanPSNR = 'Mean PSNR = '+ str(np.array(psnrs).mean()) + '\n' 
        meanSSIM = 'Mean SSIM = '+ str(np.array(ssims).mean()) + '\n'
        meanLPIPS = 'Mean LPIPS = '+ str(np.array(lpipss).mean()) + '\n'
            
        # Writing summary...
        f.write("================Summary================\n")
        f.write(meanPSNR)
        f.write(meanSSIM)
        f.write(meanLPIPS)
        f.close()
    print("Done!")
