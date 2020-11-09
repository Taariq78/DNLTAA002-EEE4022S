'''The core componets of training function was provided by - 

VoxelMorph: A Learning Framework for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231
An Unsupervised Learning Model for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
CVPR 2018. eprint arXiv:1802.02604 -

and was greatly modified to better suit the needs of the project'''

# python imports
import os
import glob
import random

# external imports
import numpy as np
import torch
from torch.optim import Adam
from voxelmorph.pytorch.model import SpatialTransformer

# internal imports
import datagenerators
import losses

def register(target, target_seg, model, moving, moving_seg, reg_param, data_loss):
          
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    spatial=SpatialTransformer((256,256))
    spatial=spatial.to(device)

    sim_loss_fn_V= losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn_V = losses.gradient_loss
    dice_loss_fn_V = losses.diceLoss                        

    # pass image-pair through model      
    warp_V, flow_V = model(moving, target)
    warp_seg_V = spatial(moving_seg,flow_V)

    # calculate validation losses      
    loss_dice_V = dice_loss_fn_V(target_seg, warp_seg_V)
    recon_loss_V = sim_loss_fn_V(warp_V, target) 
    loss_grad_V = grad_loss_fn_V(flow_V)

    V_Loss = recon_loss_V + reg_param*loss_grad_V + 0.01*loss_dice_V

    return V_Loss

def train(data_dir, 
          train_vol_names,
          atlas_file,
          lr,
          data_loss,
          reg_param, 
          batch_size,
          n_save_iter,
          model_dir, network, EPOCH, validation_vol_names, seg = True):
          
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Produce the loaded atlas
    atlas = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    # Atlas's corresponding segment
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

    # data generator for training data and validation data
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size, return_segs=True)
    validate_example_gen = datagenerators.example_gen(validation_vol_names, batch_size, return_segs=True)

    # set up atlas tensor
    atlas_vol_bs = np.repeat(atlas, batch_size, axis=0)
    input_fixed  = torch.from_numpy(atlas_vol_bs).to(device).float()
    # normalise data between 0 and 1
    input_fixed  = input_fixed/255.0
    input_fixed  = input_fixed.permute(0, 3, 1, 2)

    atlas_seg_bs = np.repeat(atlas_seg, batch_size, axis=0)
    seg_fixed  = torch.from_numpy(atlas_seg_bs).to(device).float()
    # normalise data between 0 and 1
    seg_fixed  = seg_fixed/255.0
    seg_fixed  = seg_fixed.permute(0, 3, 1, 2)

    spatial=SpatialTransformer((256,256))
    spatial=spatial.to(device)

    all_losses=torch.empty(1)
    validation = []
    model.train() 
    # Training loop.
    for epoch in range(EPOCH):   
      for i in range(len(train_vol_names)//batch_size):
          # Save model checkpoint
          if i % n_save_iter == 0:
              save_file_name = os.path.join(model_dir, '%d_%d.ckpt' % (epoch,i))
              torch.save(model.state_dict(), save_file_name)
              
              valid_loss=torch.empty(1)
              with torch.no_grad():
                model.eval()
                for data in range(len(validation_vol_names)//batch_size):
                  moving_image_V, moving_segment_V = next(validate_example_gen)  
          
                  input_moving_V = torch.from_numpy(moving_image_V[0]).to(device).float()
                  input_moving_V=input_moving_V/255.0
                  input_moving_V = input_moving_V.permute(0, 3, 1, 2)

                  seg_moving_V = torch.from_numpy(moving_segment_V[0]).to(device).float()
                  seg_moving_V=seg_moving_V/255.0
                  seg_moving_V = seg_moving_V.permute(0, 3, 1, 2)
                  
                  Total_loss = register(input_fixed, seg_fixed, model, input_moving_V, seg_moving_V, reg_param, data_loss)
                  valid_loss=torch.cat((valid_loss,torch.as_tensor(Total_loss.item()).view(1)),0)  
                #store validation loss    
                validation.append(torch.mean(valid_loss[1:]))
              model.train()           

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
          
          if seg:
                    warp_seg=spatial(seg_moving,flow)
                    dice_loss = dice_loss_fn(seg_fixed,warp_seg)
          else:
                    dice_loss = 0

          # Calculate loss
          recon_loss = sim_loss_fn(warp, input_fixed) 
          grad_loss = grad_loss_fn(flow)
          loss = recon_loss + reg_param * grad_loss + 0.01*dice_loss

          print("Epoch:%d" % (epoch))
          print("Batch_number:%d" % (i))
          print("loss(total):%f" % (loss.item()))
          print("recons_loss:%f" % (recon_loss.item()))
          print("grad_loss:%f" % (grad_loss.item()))
          print("dice_loss:%f" % (dice_loss.item()))
          print("---------------------------------------\n")
          
          #record and store loss in tensor
          all_losses=torch.cat((all_losses,torch.as_tensor(loss.item()).view(1)),0)

          # Backwards and optimize
          opt.zero_grad()
          loss.backward()
          opt.step()

    return all_losses, validation

