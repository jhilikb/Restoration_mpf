import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
import torch.nn.functional as f
# from thop import profile




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, pf, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (pf,ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, ResnetGenShared, shenc_args, shdec_args)
    else:
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD


##############################################################################
# Classes
##############################################################################
class myconvv4(nn.Module):
    def __init__(self, inc, outch, kernel_size=3,
                                stride=1, padding=1, bias=True):
        
        super().__init__()
        och4=int(outch/4)
        och2=int(outch/2)
        self.conv2d_1 = nn.Conv2d(inc, och4, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv2d_2 = nn.Conv2d(inc, och4, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.conv2d_3 = nn.Conv2d(inc, och2, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)


    def forward(self, input):
        
        x1= torch.cos(self.conv2d_1(input))
        x2= torch.sin(self.conv2d_2(input))
        x3= f.gelu(self.conv2d_3(input))
        y=torch.cat((x1,x2,x3),1)
        return y
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size,padding,bias):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding,groups=in_ch,bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
class fou(nn.Module):
    def __init__(self, inc, outch, kernel_size=5, stride=1, padding=2, bias=True):
        
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(inc*6, outch, kernel_size,padding, bias)


    def forward(self, inputa):
        input=(inputa-0.5)/0.5
        input1=torch.cat((torch.sin(input*3.14159265359),torch.cos(input*3.14159265359),torch.sin(2*input*3.14159265359),torch.cos(2*input*3.14159265359),torch.sin(4*input*3.14159265359),torch.cos(4*input*3.14159265359) ),1)
        return self.conv2d(input1)

class leg(nn.Module):
    def __init__(self, inc, outch, kernel_size=5, stride=1, padding=2, bias=True):
        
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(inc*3, outch, kernel_size,padding, bias)


    def forward(self, inputa):
        input=(inputa-0.5)/0.5
        input1=torch.cat((input,0.5*(3*input*input - 1), 0.5* ( 5*input*input*input -(3*input*input)) ),1)
        return self.conv2d(input1)

class cheby(nn.Module):
    def __init__(self, inc, outch, kernel_size=5, stride=1, padding=2, bias=True):
        
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(inc*3, outch, kernel_size,padding, bias)


    def forward(self, inputa):
        input=(inputa-0.5)/0.5
        input1=torch.cat((input,2*input*input - 1, 4*input*input*input -(3*input*input) ),1)
        return self.conv2d(input1)



# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, pf=0,ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids
        if pf ==0:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1,padding=0,bias=use_bias),
                     norm_layer(ngf),
                     nn.PReLU()]
        if pf ==1:
            model = [nn.ReflectionPad2d(3),
                     leg(input_nc, ngf, kernel_size=7, stride=1,padding=0,bias=use_bias),
                     norm_layer(ngf),
                     nn.PReLU()]
        if pf ==2:
            model = [nn.ReflectionPad2d(3),
                     cheby(input_nc, ngf, kernel_size=7, stride=1,padding=0,bias=use_bias),
                     norm_layer(ngf),
                     nn.PReLU()]
        if pf ==3:
            model = [nn.ReflectionPad2d(3),
                     fou(input_nc, ngf, kernel_size=7, stride=1,padding=0,bias=use_bias),
                     norm_layer(ngf),
                     nn.PReLU()]
        if pf ==4:
            model = [nn.ReflectionPad2d(3),
                     myconvv4(input_nc, ngf, kernel_size=7, stride=1,padding=0,bias=use_bias),
                     norm_layer(ngf),
                     nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, pf=0, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, pf=0,ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 1, kernel_size=7, padding=0),
                  nn.GELU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv1 = myconv(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x*self.sigmoid(y)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        conv_block +=  [ChannelAttention(dim)]
        #conv_block += [ SpatialAttention()]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3


class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])

        import time
        import copy
        import torch
        from thop import profile, clever_format
        
        # ==========================================
        # 1. SETUP MODELS & INPUTS
        # ==========================================
        # Detect best hardware acceleration (CUDA GPU, Apple MPS, or fallback to CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Benchmarking inference on target hardware: {device.type.upper()}")
        
        # Move models to the evaluation hardware device
        model_e = copy.deepcopy(self.encoders[0]).to(device)
        model_d = copy.deepcopy(self.decoders[0]).to(device)
        model_e.eval()
        model_d.eval()
        
        # Generate inputs directly on the target hardware device
        input_e = torch.randn(1, 3, 256, 256, device=device)
        input_d = torch.randn(1, 256, 64, 64, device=device)
        
        # ==========================================
        # 2. MEASURE COMPLEXITY (FLOPs / MACs)
        # ==========================================
        # Note: Profile runs on CPU to prevent issues with custom operator registration on GPUs
        cpu_model_e = copy.deepcopy(model_e).cpu()
        cpu_model_d = copy.deepcopy(model_d).cpu()
        cpu_input_e = input_e.cpu()
        cpu_input_d = input_d.cpu()
        
        e_macs, e_params = profile(cpu_model_e, inputs=(cpu_input_e,), verbose=False)
        d_macs, d_params = profile(cpu_model_d, inputs=(cpu_input_d,), verbose=False)
        
        e_flops = e_macs * 2
        d_flops = d_macs * 2
        
        # ==========================================
        # 3. MEASURE INFERENCE TIME (LATENCY)
        # ==========================================
        def measure_latency(model, dummy_input, device_type, num_runs=100, warmup_runs=10):
            # Warmup runs to load CUDA kernels, initialize memory, and stabilize CPU frequency
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(dummy_input)
                
                # Synchronize device clock before starting timer
                if device_type == "cuda":
                    torch.cuda.synchronize()
                elif device_type == "mps":
                    torch.mps.synchronize()
            
                start_time = time.perf_counter()
                
                # Performance execution loop
                for _ in range(num_runs):
                    _ = model(dummy_input)
                    
                # Synchronize device clock before ending timer
                if device_type == "cuda":
                    torch.cuda.synchronize()
                elif device_type == "mps":
                    torch.mps.synchronize()
                    
                end_time = time.perf_counter()
                
            avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
            return avg_latency_ms

        e_time = measure_latency(model_e, input_e, device.type)
        d_time = measure_latency(model_d, input_d, device.type)

        # ==========================================
        # 4. PRINT BENCHMARKING REPORT
        # ==========================================
        print("\n" + "="*45)
        print(f"{'BLOCK COMPLEXITY & LATENCY REPORT':^45}")
        print("="*45)
        print(f"--- ENCODER BLOCK ---")
        print(f"Parameters:     {clever_format([e_params], '%.3f')}")
        print(f"Complex MACs:   {clever_format([e_macs], '%.3f')}")
        print(f"Total FLOPs:    {clever_format([e_flops], '%.3f')}")
        print(f"Avg Latency:    {e_time:.3f} ms")
        
        print(f"\n--- DECODER BLOCK ---")
        print(f"Parameters:     {clever_format([d_params], '%.3f')}")
        print(f"Complex MACs:   {clever_format([d_macs], '%.3f')}")
        print(f"Total FLOPs:    {clever_format([d_flops], '%.3f')}")
        print(f"Avg Latency:    {d_time:.3f} ms")
        
        print(f"\n--- TOTAL COMBINED PIPELINE ---")
        print(f"Total FLOPs:    {clever_format([e_flops + d_flops], '%.3f')}")
        print(f"Total Latency:  {e_time + d_time:.3f} ms")
        print("="*45)




        
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
