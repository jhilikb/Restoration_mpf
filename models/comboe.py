import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import contextual as cl
import  torch.nn.functional as F
import torchvision
from torchvision.utils import save_image


layers = {"conv_5_2": 1.0}
def dark_channel(I, size=15):
    """Compute dark channel prior of an image"""
    # I assumed to be in range [0,1]
    min_channel, _ = I.min(dim=1, keepdim=True)

    # erosion = max-pool on negative
    pad = size // 2
    dark = -F.max_pool2d(-min_channel, size, stride=1, padding=pad)
    
    return dark

def estimate_atmospheric_light(I, dark, top_percent=0.001):
    """Estimate atmospheric light in hazy image"""
    B, _, H, W = I.shape
    num_pixels = max(int(H * W * top_percent), 1)

    dark_flat = dark.view(B, -1)
    I_flat = I.view(B, 3, -1)

    A_list = []
    for b in range(B):
        _, idx = torch.topk(dark_flat[b], num_pixels, largest=True)

        candidates = I_flat[b][:, idx]  # [3, num_pixels]
        intensities = candidates.sum(dim=0)
        best = torch.argmax(intensities)

        A_pixel = candidates[:, best]
        A_list.append(A_pixel)

    return torch.stack(A_list, dim=0)

    

def estimate_transmission(I, A, omega=0.95, size=15):
    """Estimate transmission map using DCP"""
    
   

    B, C, H, W = I.shape
    A_img = A.view(B, 3, 1, 1)

    # IMPORTANT: keep computation meaningful by using abs(A)
    normalized = I / (A_img + 1e-6)

    dark_norm = dark_channel(normalized, size)
    t = 1 - omega * dark_norm
    return t #torch.clamp(t, 0.05, 1.0)


def myk(I):
    I=torch.clamp(I,0,.999)
    dark = dark_channel(I, size=15)
    
    #print(dark.min(),dark.max())
    A = estimate_atmospheric_light(I, dark)
    #print(A)
    t = estimate_transmission(I, A)
    #print(t.min(),t.max())
    A_img = A.view(-1, 3, 1, 1)

    k = ((1.0/t) * (I - A_img) + (A_img - 1.0)) / (I - 1.0 + 1e-6)
    #p, idx = torch.topk(k, int(I.shape[2]*I.shape[3]*0.0001), largest=True)
    
    
    #k1=torch.clamp(k, k.min(), p.mean())


    B, C, H, W = I.shape

# k has the same shape as I (or at least starts with B)
# Find per-sample k values
# topk over flattened spatial + channel dimensions per sample
    k_flat = k.view(B, -1)

# number of elements to select
    num = int(C * H * W * 0.0001)

    p, idx = torch.topk(k_flat, num, largest=True)

# compute mean of the top-k per sample -> shape (B,)
    p_mean = p.mean(dim=1)

# reshape for broadcasting over (C,H,W)
    p_mean = p_mean.view(B, 1, 1, 1)

# clamp per sample
    k1 = torch.clamp(k, k.min(), p_mean)
    grid = torchvision.utils.make_grid(k1, nrow=4)  # 4 per row → 2 rows
    rd = torch.randint(0, 10_000_000, (1,)).item()

# Save to a file
    save_image(grid, "/workspace/im/k/"+str(rd)+"_grid.png",normalize=True, scale_each=True)

    
    
    return k1

def myj(img, t, t0=0.1):
    """
    Recover dehazed image J from I, t, and A for a batch of images.
    img: (B, C, H, W)
    t:   (B, 1, H, W) or (B, H, W)
    A:   broadcastable, e.g. (B, C, 1, 1) or (C,)
    """
    # Convert to float32
    #img_f = img.float()

    #img=torch.clamp(I,0,.999)
    dark = dark_channel(img, size=15)
    
    #print(dark.min(),dark.max())
    A = estimate_atmospheric_light(img, dark)
    #print(A)
    
    A_img = A.view(-1, 3, 1, 1)

    # Ensure t has shape (B, 1, H, W)
    if t.dim() == 3:
        t = t.unsqueeze(1)

    # Clamp t to avoid division by small values
    t = torch.clamp(t, min=t0)

    # Recover scene radiance
    J = (img - A_img) / t + A_img

    # Clamp to valid image range [0, 1]
    J = torch.clamp(J, 0, 1)

    return J

class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'

    def __init__(self, opt):
        super(ComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.pf, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            dev='cuda:'+str(self.gpu_ids[0])

        
            self.L1 = torch.nn.SmoothL1Loss()
            self.grad_filter = torch.cuda.FloatTensor([-1,-1,-1,-1,8,-1,-1,-1,-1]).view(1,1,3,3)
            # self.Lc = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4').to(dev)
            self.Lc = cl.ContextualLoss(layers, max_1d_size=64).to('cuda:0')
            self.downsample = torch.nn.AvgPool2d(3, stride=2)
            
            #self.downsample2 = torch.nn.AvgPool2d(2, stride=2)
            #self.downsample4 = torch.nn.AvgPool2d(4, stride=4)
            #self.downsample8 = torch.nn.AvgPool2d(8, stride=8)
            
            self.gr = lambda x: (0.299*x[:,0:1,:,:] + 0.587*x[:,1:2,:,:] + 0.114*x[:,2:3,:,:])
            #self.gr=lambda input : (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)
            self.compedge = lambda x: (torch.abs(  F.conv2d(self.gr(x), self.grad_filter, padding=1)))
            #self.compedge = lambda a1 : torch.nn.functional.conv2d(self.gr(a1), self.grad_filter, stride=2)
            self.edge_enhance = lambda inp, out: torch.mean(F.relu(self.normedge(self.compedge(inp)) - self.normedge(self.compedge(out))
    ))


            self.normedge = lambda e: e / (e.mean(dim=[2,3], keepdim=True) + 1e-6)

# multi-scale edge loss (cleaned)
            self.edgemulti = lambda y,t: (self.L1(self.normedge(self.compedge(y)), self.normedge(self.compedge(t))) +
    self.L1(self.normedge(self.compedge(F.avg_pool2d(y,2))), self.normedge(self.compedge(F.avg_pool2d(t,2)))) +
    self.L1(self.normedge(self.compedge(F.avg_pool2d(y,4))), self.normedge(self.compedge(F.avg_pool2d(t,4)))) +
    self.L1(self.normedge(self.compedge(F.avg_pool2d(y,8))), self.normedge(self.compedge(F.avg_pool2d(t,8))))
) / 4
            self.criterionCycle = self.L1
            #self.edge = lambda y,t : self.L1(self.compedge(y), self.compedge(t))
            #self.edge = lambda y,t: self.L1(self.normedge(self.compedge(y)), self.normedge(self.compedge(t)))
            
            #self.edgemulti = lambda y,t : (self.L1(self.compedge(y), self.compedge(t))+self.L1(self.compedge(self.downsample2(y)), self.compedge(self.downsample2(t)))+self.L1(self.compedge(self.downsample4(y)), self.compedge(self.downsample4(t)))+self.L1(self.compedge(self.downsample8(y)), self.compedge(self.downsample8(t))))/4
            
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
            self.criterionLatent = lambda y,t : self.L1(y, t.detach())
            
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0],f[0],v) + \
                                                networks.GANLoss(r[1],f[1],v) + \
                                                networks.GANLoss(r[2],f[2],v)) / 3
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            
            self.loss_cycle = [0]*self.n_domains
            self.loss_cyclec = [0]*self.n_domains
            self.loss_edge = [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
        self.image_paths = input['path']

    def test(self):
        with torch.no_grad():
            self.visuals = [self.real_A]
            self.labels = ['real_%d' % self.DA]

            # cache encoding to not repeat it everytime
            encoded = self.netG.encode(self.real_A, self.DA)
            for d in range(self.n_domains):
                if d == self.DA and not self.opt.autoencode:
                    continue
                tf = self.netG.decode(encoded, d)
                tf = tf[:,:,:self.real_A.shape[2],:self.real_A.shape[3]]
                fake = myj(self.real_A,tf)
                fake= F.relu(fake)
                self.visuals.append( fake )
                self.labels.append( 'fake_%d' % d )
                # if self.opt.reconstruct:
                #     rec = self.netG.forward(fake, d, self.DA)
                #     self.visuals.append( rec )
                #     self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, pred_real, fake, domain):
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)

    def backward_G(self):
        # dark = dark_channel(self.real_A, size=15)
        # A = estimate_atmospheric_light(self.real_A, dark)
        # t = estimate_transmission(self.real_A, A)
        # self.real_kA=((1/t)*(self.real_A-A) + (A-1))/(self.real_A-1)

        # dark = dark_channel(self.real_B, size=15)
        # A = estimate_atmospheric_light(self.real_B, dark)
        # t = estimate_transmission(self.real_B, A)
        # self.real_kB=((1/t)*(self.real_B-A) + (A-1))/(self.real_B-1)
        #j=k*self.real_A - k +1
        
        encoded_A = self.netG.encode(self.real_A, self.DA)
        encoded_B = self.netG.encode(self.real_B, self.DB)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:
            # Same encoder and decoder should recreate image
            idt_A = self.netG.decode(encoded_A, self.DA)
            loss_idt_A = self.criterionIdt(idt_A, self.real_A)
            idt_B = self.netG.decode(encoded_B, self.DB)
            loss_idt_B = self.criterionIdt(idt_B, self.real_B)
        else:
            loss_idt_A, loss_idt_B = 0, 0

        # GAN loss
        # D_A(G_A(A))
        k1 = self.netG.decode(encoded_A, self.DB)
        self.fake_B = F.relu(myj(self.real_A , k1 ))
        #print(self.fake_B.min(),self.fake_B.max())
        
        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake, False)
        # D_B(G_B(B))
        k2 = self.netG.decode(encoded_B, self.DA)
        self.fake_A = F.relu(myj(self.real_B,k2))
        
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake, False)
        # Forward cycle loss
        
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        k3 = self.netG.decode(rec_encoded_A, self.DA)
        self.rec_A = F.relu(myj(self.fake_B , k3 ))
        
        
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)
        self.loss_cyclec[self.DA] = self.Lc(self.rec_A, self.real_A)
        #self.loss_edge[self.DA] = self.edgemulti(self.fake_B, self.real_A)
        self.loss_edge[self.DA] = self.edgemulti(self.rec_A, self.real_A)
        # Backward cycle loss
        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        k4 = self.netG.decode(rec_encoded_B, self.DB)
        self.rec_B = F.relu(myj(self.fake_A , k4 ))
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)
        self.loss_cyclec[self.DB] = self.Lc(self.rec_B, self.real_B)
        self.loss_edge[self.DB] = self.edgemulti(self.rec_B, self.real_B)
        loss_enhance = self.edge_enhance(self.real_B, self.fake_A)

        # Optional cycle loss on encoding space
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # Optional loss on downsampled image before and after
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0

        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                 (self.loss_cyclec[self.DA] + self.loss_cyclec[self.DB]) * 0.5 + \
                 (self.loss_edge[self.DA] + self.loss_edge[self.DB]) * 0.05 + \
                 (loss_enhance)*0.05 +\
                 (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                 (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                 (loss_fwd_A + loss_fwd_B) * self.lambda_fwd
        loss_G.backward()

    def optimize_parameters(self):
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac

