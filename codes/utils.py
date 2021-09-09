import os
import torch
import numpy as np 
from torch.nn import functional as F

def get_file_name(path,suffix):
    """
    This function is used to get file name with specific suffix
    
    Parameters:
        path: path of the parent directory
        suffix: specific suffix (in the form like '.png')
    """
    name_list=[]
    file_list = os.listdir(path)
    for i in file_list:
        if os.path.splitext(i)[1] == suffix:
            name_list.append(i)
    name_list.sort()
    return name_list

def filter_events_by_key(key, x1, x2, x3, start, end): 
    """
    This function is used to filter events by the key dimension (start inclusive and end exclusive)
    e.g., new_x,new_y,new_t,new_p = filter_events_by_key(x, y, t, p, start=0, end=128) 
    returns the filted events with 0 <= x < 128
    
    Parameters:
        key: path of the parent directory
        suffix: specific suffix (in the form like '.png')
    """
    new_x1 = x1[key>=start]
    new_x2 = x2[key>=start]
    new_x3 = x3[key>=start]
    new_key = key[key>=start]
    
    new_x1 = new_x1[new_key<end]
    new_x2 = new_x2[new_key<end]
    new_x3 = new_x3[new_key<end]
    new_key = new_key[new_key<end]

    return new_key,new_x1,new_x2,new_x3


def crop(data, roiTL=(2,45), size=(256,256)):
    """
    This function is used to crop the region of interest (roi) from event frames or aps images
    
    Parameters:
        data: input data (either event frames or aps images)
        roiTL: coordinate of the top-left pixel in roi
        size: expected size of roi
    """
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
    """
    This function is used to refocus events with the predicted parameter psi
    
    Parameters:
        data: input unfocused event frames
        psi: refocusing parameter predicted by RefocusNet
        diff_t: time difference between the timestamps of event frames and the reference time
    """
    
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
    """
    This function is used to calculate MPSE in the horizontal direction
    
    Parameters:
        psi: refocusing parameter predicted by RefocusNet
        diff_t: time difference between the timestamps of event frames and the reference time
        depth: ground truth depth 
        width: width of the event frame
        v: camera moving speed in the horizontal direction
        fx: parameter from the camera intrinsic matrix
    """
    
    psi_x = psi.squeeze()[0]
    pred_depth = (2*fx*v) / (-1*psi_x*width)
    
    max_abs_diff_t = torch.max(torch.abs(diff_t))
    max_pix_shift_real = max_abs_diff_t * fx * v / depth 
    max_pix_shift_pred = max_abs_diff_t * fx * v / pred_depth
    MPSE = np.abs(max_pix_shift_real - max_pix_shift_pred)
    
    return MPSE