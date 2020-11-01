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
def train(gpu,
          data_dir,
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir, network, EPOCH):
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
    train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))

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

    return all_losses

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

