import numpy as np
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from utils import get_file_name, filter_events_by_key
import argparse

def pack_with_manual_refocus(opt, event_path, aps_path, save_name):
    """
    This function is used to manually refocus events and pack them to event frames
    
    Parameters:
        opt: basic parameters
        event_path: path of the target event data (in the form of 'npy')
        aps_path: path of the corresponding ground truth image
        save_name: saving path of the processed data
    """
    
    ## load raw data and initialize parameters ------
    data = np.load(event_path,allow_pickle=True).item()
    eventData = data.get('events')     # unfocused events
    reference_time = data.get('ref_t')  # timestamp at the reference camera pose
    fx = data.get('fx') # parameter from camera intrinsic matrix
    d = data.get('depth') # depth
    v = opt.v # camera speed (m/s)
    
    ## manual event refocusing ------
    img_size = (260,346)
    x = eventData[:,0]
    shift_x = np.round((eventData[:,2] - reference_time) * fx * v / d)
    valid_ind = (x+shift_x >= 0) * (x+shift_x < img_size[1])
    x[valid_ind] += shift_x[valid_ind]
    eventData[:,0] = x
    
    ## pack events to event frames ------
    minT = eventData[:,2].min()
    maxT = eventData[:,2].max()
    eventData[:,2] -= eventData[:,2].min()
    interval = (maxT - minT) / opt.time_step
    
    # filter events
    x = eventData[:,0]
    y = eventData[:,1]
    t = eventData[:,2]
    p = eventData[:,3]
    Xrange = (opt.roiTL[1], opt.roiTL[1] + opt.roi_size[1]) # roi
    Yrange = (opt.roiTL[0], opt.roiTL[0] + opt.roi_size[0])
    x,y,t,p = filter_events_by_key(x,y,t,p, Xrange[0], Xrange[1])
    y,x,t,p = filter_events_by_key(y,x,t,p, Yrange[0], Yrange[1])
    
    # convert events to event frames
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    T,H,W = pos.shape
    pos = pos.ravel()
    neg = neg.ravel()
    ind = (t / interval).astype(int)
    ind[ind == T] -= 1
    x = (x - Xrange[0]).astype(int)
    y = (y - Yrange[0]).astype(int)
    pos_ind = p == 1
    neg_ind = p == 0
    
    np.add.at(pos, x[pos_ind] + y[pos_ind]*W + ind[pos_ind]*W*H, 1)
    np.add.at(neg, x[neg_ind] + y[neg_ind]*W + ind[neg_ind]*W*H, 1)
    pos = np.reshape(pos, (T,H,W))
    neg = np.reshape(neg, (T,H,W))
                
    ## save event data -------       
    processed_event = dict()
    processed_event['Pos'] = pos
    processed_event['Neg'] = neg
    processed_event['size'] = opt.roi_size
    processed_event['Inter'] = interval
    processed_event['depth'] = d
    np.save(opt.save_event_path+save_name+'.npy', processed_event)
    
    ## save croped APS -------
    gt = cv2.imread(aps_path,cv2.IMREAD_GRAYSCALE)
    processed_gt = gt[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    cv2.imwrite(opt.save_aps_path+save_name+'.png', processed_gt)   
    
def pack_without_refocus(opt, event_path, aps_path, save_name):
    """
    This function is used to pack unfocused events to event frames
    
    Parameters:
        opt: basic parameters
        event_path: path of the target event data (in the form of 'npy')
        aps_path: path of the corresponding ground truth image
        save_name: saving path of the processed data
    """
    
    ## load raw data and initialize parameters ------
    data = np.load(event_path,allow_pickle=True).item()
    eventData = data.get('events')  # unfocused events
    d = data.get('depth') # depth
    fx = data.get('fx') # parameter from camera intrinsic matrix
    ref_t = data.get('ref_t') # timestamp at the reference camera pose
    
    ## pack events ------
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    minT = eventData[:,2].min()
    maxT = eventData[:,2].max()
    ref_t -= minT
    eventData[:,2] -= minT
    interval = (maxT - minT) / opt.time_step
    index_t = []
    for i in range(opt.time_step):
        index_t.append(i*interval)
    index_t = np.array(index_t)
    
    # filter events
    x = eventData[:,0]
    y = eventData[:,1]
    t = eventData[:,2]
    p = eventData[:,3]
    Xrange = (opt.roiTL[1], opt.roiTL[1] + opt.roi_size[1])
    Yrange = (opt.roiTL[0], opt.roiTL[0] + opt.roi_size[0])
    x,y,t,p = filter_events_by_key(x,y,t,p, Xrange[0], Xrange[1])
    y,x,t,p = filter_events_by_key(y,x,t,p, Yrange[0], Yrange[1])
    
    # convert events to event frames
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    T,H,W = pos.shape
    pos = pos.ravel()
    neg = neg.ravel()
    ind = (t / interval).astype(int)
    ind[ind == T] -= 1
    x = (x - Xrange[0]).astype(int)
    y = (y - Yrange[0]).astype(int)
    pos_ind = p == 1
    neg_ind = p == 0
    np.add.at(pos, x[pos_ind] + y[pos_ind]*W + ind[pos_ind]*W*H, 1)
    np.add.at(neg, x[neg_ind] + y[neg_ind]*W + ind[neg_ind]*W*H, 1)
    pos = np.reshape(pos, (T,H,W))
    neg = np.reshape(neg, (T,H,W))
    
    ## save event data -------            
    processed_event = dict()
    processed_event['Pos'] = pos # positive event frame 
    processed_event['Neg'] = neg # negative event frame
    processed_event['size'] = opt.roi_size # frame size
    processed_event['depth'] = d # target depth
    processed_event['Inter'] = interval # time interval of one event frame
    processed_event['ref_t'] = ref_t # timestamp of reference camera position
    processed_event['fx'] = fx # camera intrinsic parameter
    processed_event['roiTL'] = opt.roiTL # top-left coordinate of roi
    processed_event['index_t'] = index_t # timestamp of each event frame
    np.save(opt.save_event_path+save_name+'.npy', processed_event)

    ## save APS -------
    gt = cv2.imread(aps_path, cv2.IMREAD_GRAYSCALE)
    processed_gt = gt[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    cv2.imwrite(opt.save_aps_path+save_name+'.png', processed_gt)    


if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="preprocess data + event refocusing")
    parser.add_argument("--input_event_path", type=str, default="./Example_data/Raw/Event/", help="input event path")
    parser.add_argument("--input_aps_path", type=str, default="./Example_data/Raw/APS/", help="input aps path")
    parser.add_argument("--save_event_path", type=str, default="./Example_data/Processed/Event/", help="saving event path")
    parser.add_argument("--save_aps_path", type=str, default="./Example_data/Processed/APS/", help="saving aps path")
    parser.add_argument("--do_event_refocus", type=int, default=1, help="manually refocus events (1) or not (0)")
    parser.add_argument("--time_step", type=int, default=30, help="the number of event frames ('N' in paper)")
    parser.add_argument("--roiTL", type=tuple, default=(0,0), help="coordinate of the top-left pixel in roi")
    parser.add_argument("--roi_size", type=tuple, default=(260,346), help="size of roi")
    parser.add_argument("--v", type=float, default=0.177, help="default camera speed (m/s) ")
    opt = parser.parse_args()
    
    ## obtain input file names 
    input_event_name = get_file_name(opt.input_event_path,'.npy')  
    input_event_name.sort()
    input_aps_name = get_file_name(opt.input_aps_path,'.png')  
    input_aps_name.sort()
    
    ## create directory
    if not os.path.exists(opt.save_event_path):
        os.mkdir(opt.save_event_path)
    if not os.path.exists(opt.save_aps_path):
        os.mkdir(opt.save_aps_path)
    
    if opt.do_event_refocus == 1:
        print("Pack events with manual refocusing ...")
    else:
        print("Pack events without refocusing ...")
    
    ## preprocessing 
    for i in range(len(input_event_name)):
        print('processing event ' + input_event_name[i] + '...')
        current_event_path = os.path.join(opt.input_event_path, input_event_name[i])
        current_aps_path = os.path.join(opt.input_aps_path, input_aps_name[i])
        save_name = input_event_name[i][:-4]
        if opt.do_event_refocus == 1:
            pack_with_manual_refocus(opt, current_event_path, current_aps_path, save_name)
        else:
            pack_without_refocus(opt, current_event_path, current_aps_path, save_name)
    print('Completed!')
