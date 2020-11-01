def register(target, moving, network, init_model_file):

    import losses

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

    sim_loss_MSE = losses.mse_loss
    sim_loss_CC = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    dice_loss_fn = losses.diceLoss
    dice_hard = dice

    spatial=SpatialTransformer((256,256))
    spatial=spatial.to(device)

    # Set up model
    model = network
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    initial_loss_MSE = sim_loss_MSE(input_moving, input_fixed) 
    initial_loss_CC = sim_loss_CC(input_moving, input_fixed) 

    fixed_seg = input_fixed_seg.to('cpu')
    fixed_seg = fixed_seg.detach().numpy()
    moving_seg = seg_moving.to('cpu')
    moving_seg = moving_seg.detach().numpy()

    intial_dice_hard = dice_hard(moving_seg, fixed_seg, [1]) 

    warp, flow = model(input_moving, input_fixed)

    recon_loss_MSE = sim_loss_MSE(warp, input_fixed) 
    recon_loss_CC = sim_loss_CC(warp, input_fixed) 
    loss_grad = 1*grad_loss_fn(flow)

    warp_seg = spatial(seg_moving,flow)
    loss_dice = 0.01*dice_loss_fn(input_fixed_seg, warp_seg)

    warp_seg = warp_seg.to('cpu')
    warp_seg = warp_seg.detach().numpy()


    dice_score = dice_hard(warp_seg, fixed_seg, [1])   
    Total_Loss = recon_loss_CC + loss_grad + loss_dice
#    dice_loss = dice_loss_fn(input_fixed_seg, warp_seg)

    return Total_Loss
