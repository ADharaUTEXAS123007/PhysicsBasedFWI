"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
import deepwave
import numpy as np
import sys
sys.path.append('./models')
sys.path.append('/disk/student/adhara/WORK/DeniseFWI/virginFWI/DENISE-Black-Edition/')
import pyapi_denise as api
import os

def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total


def eval_loss2(net, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss = 0
    total = 1
    A_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainA/1.npy')
    A_img = np.expand_dims(A_img,0)
    A = torch.from_numpy(A_img)
    A = A.float()
    
    B_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainB/1.npy')
    B_img = np.expand_dims(B_img,0)
    B_img = np.expand_dims(B_img,0)
    B = torch.from_numpy(B_img)
    B = B.float()
    
    C_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Simple/trainC/1.npy')
    C_img = np.expand_dims(C_img,0)
    C_img = np.expand_dims(C_img,0)
    C = torch.from_numpy(C_img)
    C = C.float()
    
    print("shape of A :", np.shape(A))
    latent = torch.ones(1,1,1,1)
    lstart = 1
    epoch1 = 2
    [fake_B,grad,latent,loss_D_MSE,down3,up2,up1] = net(B,A,lstart,epoch1,latent,C)
    
    print("loss D MSE :", loss_D_MSE)
    
    return loss_D_MSE

def eval_loss3(net, ind, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss = 0
    total = 1
    A_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainA/1.npy')
    A_img = np.expand_dims(A_img,0)
    A = torch.from_numpy(A_img)
    A = A.float()
    
    B_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainB/1.npy')
    B_img = np.expand_dims(B_img,0)
    #B_img = np.expand_dims(B_img,0)
    B = torch.from_numpy(B_img)
    B = B.float()
    B = B/10.0
    
    C_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainC/1.npy')
    C_img = np.expand_dims(C_img,0)
    #C_img = np.expand_dims(C_img,0)
    C = torch.from_numpy(C_img)
    C = C.float()
    C = C/10.0

    D_img = np.load('/disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/trainD/1.npy')
    D_img = np.expand_dims(D_img,0)
    #D_img = np.expand_dims(D_img,0)
    D = torch.from_numpy(D_img)
    D = D.float()
    
    print("shape of A :", np.shape(A))
    latent = torch.ones(1,1,1,1)
    lstart = 1
    epoch1 = 2
    freq = 20
    #[fake_B,grad,latent,loss_D_MSE,down3,up2,up1] = net(B,A,lstart,epoch1,latent,C)
    print("shape of A :", np.shape(A))
    print("shape of B :", np.shape(B))
    print("shape of C :", np.shape(C))
    print("shape of D :", np.shape(D))


    [fake_Vp,fake_Vs,fake_Rho, grad,latent,vp_grad,vs_grad,rho_grad,loss_D_MSE] = net(B,A,lstart,epoch1,latent,C,D,freq,1,ind)  # G(A)
    #loss_D_MSE = 0
    print("loss D MSE :", loss_D_MSE)
    loss_D_MSE = loss_D_MSE*(10**5)
    
    return loss_D_MSE


def eval_loss4(model, ind, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    total_loss = 0
    total = 1

    vpst = model[0,:]
    vsst = model[1,:]
    rhost = model[2,:]
    

    denise_root = '/disk/student/adhara/WORK/DeniseFWI/virginFWI/DENISE-Black-Edition/'
    d = api.Denise(denise_root, verbose=1)
    d.save_folder = '/disk/student/adhara/DOUTPUTS/'
    d.set_paths()
        
        #model = api.Model(vp, vs, rho, dx)
        #print(model)
        
    # Receivers
    drec = int(10.)   #simple_model
    depth_rec = int(100.)  # receiver depth [m]
    ######depth_rec = 80. #simple_model
    xrec1 = int(390.)     # 1st receiver position [m]
    ######xrec1 = 100.
    xrec2 = int(2610.)     # last receiver position [m]
    #####xrec2 = 1700.
    xrec = np.arange(xrec1, xrec2 + dx, drec)
    yrec = depth_rec * (xrec / xrec)

    # Sources
    dsrc = int(80.) # source spacing [m]
    #######dsrc = 120.
    depth_src = int(20.)  # source depth [m]
    #######depth_src = 40.
    xsrc1 = int(390.) # 1st source position [m]
    ######xsrc1 = 100.
    xsrc2 = int(2610.) # last source position [m]
    #######xsrc2 = 1700.
    xsrcoriginal = np.arange(xsrc1, xsrc2 + dx, dsrc)
    #xsrc = xsrcoriginal[idx[0:6]]
    xsrc = xsrcoriginal
    ysrc = depth_src * xsrc / xsrc
    tshots = len(xsrc)

    # Wrap into api
    fsource = 10.0
    rec = api.Receivers(xrec, yrec)
    src = api.Sources(xsrc, ysrc, fsource)

    d.ITERMAX = 1
    d.verbose = 0
    print("shape of vp :", np.shape(vp))
    print("shape of vs :", np.shape(vs))
    print("shape of rho :", np.shape(rho))
    print("shape of xsrc :", np.shape(xsrc))

    print(f'NSRC:\t{len(src)}')
    print(f'NREC:\t{len(rec)}')
    d.NPROCX = 6
    d.NPROCY = 6
    d.PHYSICS = 1
    d.TIME = 5.0
        #d.NT = 2.5e-03
        #d.VPUPPERLIM = 3000.0
        #d.VPLOWERLIM = 1500.0
        #d.VSUPPERLIM = 1732.0
        #d.VSLOWERLIM = 866.0
        #d.RHOUPPERLIM = 2294.0
        #d.RHOLOWERLIM = 1929.0
        
    d.VPUPPERLIM = 3000.0
    d.VPLOWERLIM = 1500.0
    d.VSUPPERLIM = 1732.0
    d.VSLOWERLIM = 866.0
    d.RHOUPPERLIM = 2294.0
    d.RHOLOWERLIM = 1829.0
    d.SWS_TAPER_GRAD_HOR = 0.0

    model_init = api.Model(vpst, vsst, rhost, dx)

    d.fwi_stages = []
        #d.add_fwi_stage(fc_low=0.0, fc_high=20.0)
        #d.add_fwi_stage(fc_low=0.0, fc_high=20.0)
        #for i, freq in enumerate([20]
        #d.add_fwi_stage(fc_low=0.0, fc_high=int(epoch1/10)+1.0)
        #d.add_fwi_stage(fc_low=0.0, fc_high=30.0)
    d.add_fwi_stage(fc_high=10)

    print(f'Stage {0}:\n\t{d.fwi_stages[0]}\n')
            
        #print(f'Stage {0}:\n\t{d.fwi_stages[0]}\n')
    os.system('rm -rf loss_curve_grad.out')
    
    print(f'Target data: {d.DATA_DIR}')
        ###d.grad(model_init, src, rec)
    d.forward(model_init,src,rec)

    [shots_y, file_y] = d.get_shots(keys=['_y'],return_filenames=True)
    [shots_x, file_x] = d.get_shots(keys=['_x'],return_filenames=True)

    [shots_y_ob, file_y_ob] = d.get_observed_shots(keys=['_y'],return_filenames=True)
    [shots_x_ob, file_x_ob] = d.get_observed_shots(keys=['_x'],return_filenames=True)


    shots = np.concatenate((shots_y,shots_x))
    shots_obs = np.concatenate((shots_y_ob,shots_x_ob))

    loss = np.linalg.norm(shots-shots_obs)

    #loss_D_MSE = 0
    print("loss D MSE :", loss_D_MSE)
    loss_D_MSE = loss*(10**5)
    
    return loss_D_MSE
