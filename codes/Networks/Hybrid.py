import torch
import torch.nn as nn
from Networks.networks import  define_Res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.75 # decay constants 

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()  

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) 
    return mem, spike

# cofiguration for spiking layers
cfg_snn =  [(2, 8, 1, 0, 1),
           (8, 16, 1, 1, 3),
           (16+8, 32, 1, 3, 7)] 


class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        
        ## Define SNN encoder
        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[0]
        self.conv1=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
 
        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[1]
        self.conv2=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
 
        in_planes, out_planes, stride, padding, kernel_size = cfg_snn[2]
        self.conv3=nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=kernel_size,stride=stride,padding=padding)
        
        ## Define CNN decoder 
        self.Gen = define_Res(32, 1, 64, 15,norm='batch',init_type='kaiming', use_dropout=True) 


    def forward(self, input, time_window = 20):
        ## initialization
        batch_size = input.shape[0]
        inpsize = (input.shape[3],input.shape[4])
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_snn[0][1], inpsize[0], inpsize[1], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_snn[1][1], inpsize[0], inpsize[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_snn[2][1], inpsize[0], inpsize[1], device=device)
        sumspike = torch.zeros(batch_size, cfg_snn[2][1], inpsize[0], inpsize[1], device=device)
        
        ## SNN encoder
        for step in range(time_window): # simulation time steps
            inp = input[:,step,:]
            x = inp
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = c1_spike
            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem,c2_spike)
            x = torch.cat((c2_spike, c1_spike), 1)
            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem,c3_spike)
            sumspike += c3_spike 
        # normalize SNN output
        x = sumspike / time_window
 
        ## CNN decoder
        outputs = self.Gen(x) 
        
        return outputs