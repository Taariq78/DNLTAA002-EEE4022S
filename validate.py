"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
#from model import cvpr2018_net
import datagenerators
import losses
from voxelmorph.pytorch.model import SpatialTransformer

def register(target, moving, network, init_model_file):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    atlas_t = np.load(target)
    atlas_vol_t = atlas_t['vol'][np.newaxis, ..., np.newaxis]
    atlas_seg_t = atlas_t['seg'][np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol_t.shape[1:-1]
    
    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol_t).to(device).float()
    input_fixed  = input_fixed/255.0
    input_fixed  = input_fixed.permute(0, 3, 1, 2)

    input_fixed_seg  = torch.from_numpy(atlas_seg_t).to(device).float()
    input_fixed_seg  = input_fixed_seg/255.0
    input_fixed_seg  = input_fixed_seg.permute(0, 3, 1, 2)
    
    atlas_m = np.load(moving)
    atlas_vol_m = atlas_m['vol_data'][np.newaxis, ..., np.newaxis]
    atlas_seg_m = atlas_m['seg'][np.newaxis, ..., np.newaxis]
    
    input_moving = torch.from_numpy(atlas_vol_m).to(device).float()
    input_moving = input_moving/255.0
    input_moving = input_moving.permute(0, 3, 1, 2)

    seg_moving = torch.from_numpy(atlas_seg_m).to(device).float()
    seg_moving = seg_moving/255.0
    seg_moving = seg_moving.permute(0, 3, 1, 2)    

    sim_loss_MSE_V = losses.mse_loss
    sim_loss_CC_V = losses.ncc_loss
    grad_loss_fn_V = losses.gradient_loss
    dice_loss_fn_V = losses.diceLoss

    spatial=SpatialTransformer((256,256))
    spatial=spatial.to(device)

    # Set up model
    model_V = network
    model_V.to(device)
    model_V.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))


    fixed_seg = input_fixed_seg.to('cpu')
    fixed_seg = fixed_seg.detach().numpy()
    moving_seg = seg_moving.to('cpu')
    moving_seg = moving_seg.detach().numpy()

    warp, flow = model_V(input_moving, input_fixed)

    recon_loss_MSE_V = sim_loss_MSE_V(warp, input_fixed) 
    recon_loss_CC_V = sim_loss_CC_V(warp, input_fixed) 
    loss_grad_V = 0.01*grad_loss_fn_V(flow)

    warp_seg_V = spatial(seg_moving,flow)
    loss_dice_V = 0.01*dice_loss_fn_V(input_fixed_seg, warp_seg_V)

    warp_seg_V = warp_seg_V.to('cpu')
    warp_seg_V = warp_seg_V.detach().numpy()


    V_Loss = recon_loss_CC_V + loss_grad_V + loss_dice_V

    return V_Loss

def train(gpu,
          train_vol_names,
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir, network, valid, EPOCH):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Produce the loaded atlas with dims.:160x192x224.
    atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1]

    atlas_seg = np.load(atlas_file)['seg'][np.newaxis, ..., np.newaxis]

    # Get all the names of the training data
    #train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))

    model = network
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    dice_loss_fn = losses.diceLoss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size, return_segs=True )

    # set up atlas tensor
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    input_fixed  = torch.from_numpy(atlas_vol_bs).to(device).float()
    input_fixed  = input_fixed/255.0
    input_fixed  = input_fixed.permute(0, 3, 1, 2)

    atlas_seg_bs = np.repeat(atlas_seg, batch_size, axis=0)
    seg_fixed  = torch.from_numpy(atlas_seg_bs).to(device).float()
    seg_fixed  = seg_fixed/255.0
    seg_fixed  = seg_fixed.permute(0, 3, 1, 2)

    spatial=SpatialTransformer((256,256))
    spatial=spatial.to(device)

    valid_loss=torch.empty(1)
    validation = []
    all_losses=torch.empty(1)
    # Training loop.
    for epoch in range(EPOCH):
   
      for i in range(len(train_vol_names)//batch_size):

          # Save model checkpoint
          if i % n_save_iter == 0:
              save_file_name = os.path.join(model_dir, '%d_%d.ckpt' % (epoch,i))
              torch.save(model.state_dict(), save_file_name)

          # Generate the moving images and convert them to tensors.
          moving_image, moving_segment = next(train_example_gen)

          input_moving = torch.from_numpy(moving_image[0]).to(device).float()
          input_moving=input_moving/255.0
          input_moving = input_moving.permute(0, 3, 1, 2)

          seg_moving = torch.from_numpy(moving_segment[0]).to(device).float()
          seg_moving=seg_moving/255.0
          seg_moving = seg_moving.permute(0, 3, 1, 2)

          # Run the data through the model to produce warp and flow field
          warp, flow = model(input_moving, input_fixed)

          warp_seg=spatial(seg_moving,flow)

          # Calculate loss
          dice_loss = dice_loss_fn(seg_fixed,warp_seg) 
          recon_loss = sim_loss_fn(warp, input_fixed) 
          grad_loss = grad_loss_fn(flow)
          loss = recon_loss + reg_param * grad_loss + 0.01*dice_loss

          #print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)
          print("Epoch:%d" % (epoch))
          print("Batch_number:%d" % (i))
          print("loss(total):%f" % (loss.item()))
          print("recons_loss:%f" % (recon_loss.item()))
          print("grad_loss:%f" % (grad_loss.item()))
          print("dice_loss:%f" % (dice_loss.item()))
          print("---------------------------------------\n")
          
          all_losses=torch.cat((all_losses,torch.as_tensor(loss.item()).view(1)),0)            

          # Backwards and optimize
          opt.zero_grad()
          loss.backward()
          opt.step()

      for data in (valid):
        init_model_file='/content/drive/My Drive/2020/Thesis/Data/validation_0/' + str(epoch)+'_500.ckpt'
        Total_loss = register(atlas_file, data, model, init_model_file)
        valid_loss=torch.cat((valid_loss,torch.as_tensor(Total_loss.item()).view(1)),0)   
      validation.append(torch.mean(valid_loss))         



    return all_losses, validation

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--data_dir",
                        type=str,
                        help="data folder with training vols")

    parser.add_argument("--atlas_file",
                        type=str,
                        dest="atlas_file",
                        default='../data/atlas_norm.npz',
                        help="gpu id number")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=150000,
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='ncc',
                        help="data_loss: mse of ncc")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm2',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--lambda", 
                        type=float,
                        dest="reg_param", 
                        default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")

    parser.add_argument("--batch_size", 
                        type=int,
                        dest="batch_size", 
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=500,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='./models/',
                        help="models folder")


    train(**vars(parser.parse_args()))

