import os
import argparse
import numpy as np
from utils import get_file_name, filter_events_by_key

def pack_with_manual_refocus(opt, event_path, save_name):
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
    reference_time = data.get('occ_free_aps_ts')  # timestamp at the reference camera pose, which is equivalent to the timestamp of occlusion-free APS
    fx = data.get('fx') # parameter from camera intrinsic matrix
    d = data.get('depth') # depth
    v = data.get('v') # camera speed (m/s)
    img_size = data.get('size') # image size
    occ_free_aps = data.get('occ_free_aps') # occlusion-free aps
    
    ## manual event refocusing ------
    x = eventData['x'].astype(float)
    y = eventData['y'].astype(float)
    t = eventData['t'].astype(float)
    p = eventData['p'].astype(float)
    shift_x = np.round((t - reference_time) * fx * v / d)
    valid_ind = (x+shift_x >= 0) * (x+shift_x < img_size[1])
    x[valid_ind] += shift_x[valid_ind]
    
    ## pack events to event frames ------
    minT = t.min()
    maxT = t.max()
    t -= minT
    interval = (maxT - minT) / opt.time_step
    
    # filter events
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
                
    # crop occlusion-free aps according to roi 
    occ_free_aps = occ_free_aps[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    
    ## save processed data -------       
    processed_data = dict()
    processed_data['Pos'] = pos.astype(np.int16) # positive event frame 
    processed_data['Neg'] = neg.astype(np.int16) # negative event frame
    processed_data['size'] = opt.roi_size # frame size
    processed_data['Inter'] = interval # time interval of one event frame
    processed_data['depth'] = d # target depth
    processed_data['occ_free_aps'] = occ_free_aps # croped occlusion-free aps
    np.save(opt.save_path+save_name+'.npy', processed_data)
    
    
def pack_without_refocus(opt, event_path, save_name):
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
    ref_t = data.get('occ_free_aps_ts') # timestamp at the reference camera pose, which is equivalent to the timestamp of occlusion-free APS
    occ_free_aps = data.get('occ_free_aps') # occlusion-free aps
    
    ## pack events ------
    x = eventData['x']
    y = eventData['y']
    t = eventData['t']
    p = eventData['p']
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    minT = t.min()
    maxT = t.max()
    ref_t -= minT
    t -= minT
    interval = (maxT - minT) / opt.time_step
    index_t = []
    for i in range(opt.time_step):
        index_t.append(i*interval)
    index_t = np.array(index_t)
    
    # filter events
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
    
    # crop occlusion-free aps according to roi 
    occ_free_aps = occ_free_aps[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    
    ## save event data -------            
    processed_data = dict()
    processed_data['Pos'] = pos.astype(np.int16) # positive event frame 
    processed_data['Neg'] = neg.astype(np.int16) # negative event frame
    processed_data['size'] = opt.roi_size # frame size
    processed_data['depth'] = d # target depth
    processed_data['Inter'] = interval # time interval of one event frame
    processed_data['ref_t'] = ref_t # timestamp of reference camera position
    processed_data['fx'] = fx # camera intrinsic parameter
    processed_data['roiTL'] = opt.roiTL # top-left coordinate of roi
    processed_data['index_t'] = index_t # timestamp of each event frame
    processed_data['occ_free_aps'] = occ_free_aps # croped occlusion-free aps
    np.save(opt.save_path+save_name+'.npy', processed_data)


if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="preprocess data + event refocusing")
    parser.add_argument("--input_path", type=str, default="./Example_data/Raw/Event/", help="input data path")
    parser.add_argument("--save_path", type=str, default="./Example_data/Processed/Event/", help="saving data path")
    parser.add_argument("--do_event_refocus", type=int, default=1, help="manually refocus events (1) or not (0)")
    parser.add_argument("--time_step", type=int, default=30, help="the number of event frames ('N' in paper)")
    parser.add_argument("--roiTL", type=tuple, default=(0,0), help="coordinate of the top-left pixel in roi")
    parser.add_argument("--roi_size", type=tuple, default=(260,346), help="size of roi")
    opt = parser.parse_args()
    
    ## obtain input file names 
    input_data_name = get_file_name(opt.input_path,'.npy')  
    input_data_name.sort()
    
    ## create directory
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    
    if opt.do_event_refocus == 1:
        print("Pack events with manual refocusing ...")
    else:
        print("Pack events without refocusing ...")
    
    ## preprocessing 
    for i in range(len(input_data_name)):
        print('processing event ' + input_data_name[i] + '...')
        current_data_path = os.path.join(opt.input_path, input_data_name[i])
        save_name = input_data_name[i][:-4]
        if opt.do_event_refocus == 1:
            pack_with_manual_refocus(opt, current_data_path, save_name)
        else:
            pack_without_refocus(opt, current_data_path, save_name)
    print('Completed!')
