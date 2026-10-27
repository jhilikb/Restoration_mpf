
# 1.   Estimation block
# cp models/combogan_model_cycon.py    models/combogan_model.py   #K estimation
# cp models/combo1.py    models/combogan_model.py   #T estimation

# 2. Loss functions   #to be run with T estimation model along with identity, latent and contextual losses
#cp  models/combof     /models/combogan_model.py #fourier loss
#cp  models/comboe    /models/combogan_model.py  # edge loss
#cp  models/combogc    /models/combogan_model.py # geometric consistency loss
#cp  models/combolpip    /models/combogan_model.py #perceptual loss

# 3. Model

# cp models/in_t_f.py     models/networks.py   #K estimation with priors
# cp models/ckang_spdkang.py    models/networks.py   #K estimation
# cp models/net.py     models/networks.py    #T estimation


# Run training
# You can set number of residual blocks as 9 or any other
# pf indicates the kind of polynomial expansion 0 is none, 1 is legendre, 2 is chebychev, 3 is fourier 
python3 train.py --name tm  --dataroot ./dataset --n_domains 2 --niter 50 --niter_decay 50 --netG_n_blocks 9 --pf 0



# python3  test.py   --phase test   --serial_test   --name $fol    --dataroot $daim    --n_domains 2    --which_epoch 100 --resize_or_crop none    --pf 3    --netG_n_blocks 9

