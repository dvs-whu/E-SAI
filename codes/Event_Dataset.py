import os 
import torch
import numpy as np
from utils import get_file_name
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

class TestSet_ManualRefocus(Dataset): 
    """
    Test dataset for E-SAI with manual refocusing, i.e., E-SAI+Hybrid (M).
    """
    def __init__(self, path_dir): 
        # path_dir: directory path of test data
        self.path_dir = path_dir 
        self.event_names = get_file_name(self.path_dir,'.npy')
        self.event_names.sort()
    
    def __len__(self):
        return len(self.event_names)
    
    def __getitem__(self, index):
        # load test data
        event_name = self.event_names[index]
        event_path = os.path.join(self.path_dir, event_name)
        event_data = np.load(event_path, allow_pickle=True).item()
        
        # cat positive and negative events
        pos = event_data['Pos']
        neg = event_data['Neg']
        pos = torch.FloatTensor(np.expand_dims(pos,axis=1))
        neg = torch.FloatTensor(np.expand_dims(neg,axis=1))
        event_input = torch.cat((pos,neg), 1)
        
        # load occlusion-free image
        occ_free_aps = event_data['occ_free_aps']
                
        return event_input, occ_free_aps

class TestSet_AutoRefocus(Dataset):
    """
    Test dataset for E-SAI with auto refocusing, i.e., E-SAI+Hybrid (A).
    """
    def __init__(self, PathDir): 
        # path_dir: directory path of test data
        self.PathDir = PathDir
        self.data_names = get_file_name(self.PathDir,'.npy') 
        self.data_names.sort()
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self,index):
        # load test data
        event_name = self.data_names[index]
        event_path = os.path.join(self.PathDir, event_name)
        event_data = np.load(event_path, allow_pickle=True).item()
        
        # extract information
        pos = event_data['Pos']
        neg = event_data['Neg']
        diff_t = torch.FloatTensor(np.array(event_data['index_t'] - event_data['ref_t']))
        fx = event_data['fx']
        depth = event_data['depth']
        occ_free_aps = event_data['occ_free_aps']
        
        # cat positive and negative events
        pos = np.expand_dims(pos,axis=1)
        neg = np.expand_dims(neg,axis=1)
        event_input = torch.FloatTensor(np.concatenate((pos, neg), axis=1)) # size = (step, channel, H, W)
        
        return event_input, diff_t, depth, fx, occ_free_aps
        