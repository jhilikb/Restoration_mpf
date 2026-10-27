1. 3 different versions i.e. K estimation with priors, K estimation without priors and estimation are provided.  
2. Run the training script "train.sh" Uncomment the appropriate estimation block in "1" and the loss file in "2"  
3. Set the training options in train_options.py. You can use latent, identity losses,geometric losses, edge loss etc.  
4. Select the appropriate model in "3"  
5. Run the training using "sh train.sh"
6. The clean images has to be placed in dataset/train0 while the noisy images are to be placed in dataset/train1.
  
The CycleGAN framework used for training is adapted from  "https://github.com/AAnoosheh/ToDayGAN/"  
