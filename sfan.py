import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from einops import rearrange


def make_model(args, parent=False):
    return MOENet()

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size,num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()
    
    def forward(self, x):
        # Flatten the input tensor
        x = self.gap(x)+self.gap2(x)
        x = x.view(-1, self.input_size)
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x = self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)

        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')

        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)

        return gating_coeffs

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Focal(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=3, focal_factor=2, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                ) 

            self.kernel_sizes.append(kernel_size)      

    def forward(self, ctx, gates):   
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        out = ctx_all

        return out

class SpaExpert(nn.Module):
    def __init__(self, dim, focal_level=3, is_down=False):
        super(SpaExpert, self).__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.is_down = is_down

        self.focal = Focal(dim)

        if self.is_down:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2) # down scale 2

    def forward(self, x_v, x_q, gates):
        if self.is_down:
            x_v = self.down(x_v)
            gates = self.down(gates)
            x_focal = self.focal(x_v, gates)
            x_focal = F.interpolate(x_focal, scale_factor=2, mode='bilinear', align_corners=True)
            x_out   = x_q*x_focal
        else:
            x_focal = self.focal(x_v, gates)
            x_out   = x_q*x_focal
        return x_out
     

class ChannelExpert(nn.Module):
    def __init__(self, dim, focal_level=3, reduction=2):
        super(ChannelExpert, self).__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.compress1 =  nn.Conv2d(dim, dim//reduction, 1, 1, 0)
        self.compress2 =  nn.Conv2d(dim, dim//reduction, 1, 1, 0)
        self.extend  = nn.Conv2d(dim//reduction, dim, 1, 1, 0)
        self.focal = Focal(dim//reduction)

    def forward(self, x_v, x_q, gates):
        x_q = self.compress1(x_q)
        x_v = self.compress2(x_v)
        x_focal = self.focal(x_v, gates)
        x_out = x_q*x_focal
        x_out = self.extend(x_out)
        return x_out

class GatedFFN(nn.Module):
    def __init__(self, 
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio
        
        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )
        
        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2, 
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x
    
    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        
        gate = self.gate(gate)
        x = x * gate
        
        x = self.fn_2(x)
        return x

# Mixture of Expert Block (MEB)
class MEB(nn.Module):
    def __init__(self, in_size, out_size, focal_level=3, num_experts=6, k=3):
        super(MEB, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_experts = num_experts
        self.focal_level = focal_level
        self.k = k

        if self.in_size != self.out_size:
            self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.gate = GateNetwork(out_size, self.num_experts, self.k)

        self.proj1 = nn.Conv2d(out_size, 2*out_size + (self.focal_level+1), 1, 1, 0)
        self.proj2 = nn.Conv2d(out_size, out_size, 1, 1, 0)

        self.layer_1_norm = LayerNorm2d(out_size)
        self.expert_networks = nn.ModuleList([
            SpaExpert(out_size, focal_level, is_down=False),
            SpaExpert(out_size, focal_level, is_down=True),
            ChannelExpert(out_size, focal_level, reduction=2),
            ChannelExpert(out_size, focal_level, reduction=4),
            ChannelExpert(out_size, focal_level, reduction=8),
            ChannelExpert(out_size, focal_level, reduction=16),
        ])

        self.layer_2 = nn.Sequential(*[
            LayerNorm2d(out_size),
            GatedFFN(out_size, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        ])
        
    def forward(self, x):
        if self.in_size != self.out_size:
            x = self.identity(x)
        x_norm = self.layer_1_norm(x)
        C = x_norm.shape[1]
        # =======================================
        x_proj = self.proj1(x_norm)
        x_q, x_v, gates = torch.split(x_proj, (C, C, self.focal_level+1), 1)
        cof = self.gate(x_q)

        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if torch.all(cof[:,idx] == 0):
                continue
            mask = torch.where(cof[:,idx]>0)[0]
            expert_layer = self.expert_networks[idx]
            expert_out = expert_layer(x_v[mask], x_q[mask], gates[mask])
            cof_k = cof[mask,idx].view(-1,1,1,1)
            out[mask] += expert_out*cof_k
        # =======================================   
        x1 = self.proj2(out) + x

        x2 = self.layer_2(x1) + x1
        return x2

# Mixture of Fusion Block (MoFE)
class MoFE(nn.Module):
    def __init__(self, dim, num_experts=4, k=2):
        super(MoFE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.pre_fuse = nn.Conv2d(dim*2, dim, 1, 1, 0)
        self.gate = GateNetwork(dim, self.num_experts, self.k)
        self.expert_networks = nn.ModuleList([
            nn.Sequential(*[nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)]) for i in range(self.num_experts)]
        )
        
    def forward(self, x):
        x = self.pre_fuse(x)
        cof = self.gate(x)

        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if torch.all(cof[:,idx] == 0):
                continue
            mask = torch.where(cof[:,idx]>0)[0]
            expert_layer = self.expert_networks[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask,idx].view(-1,1,1,1)
            out[mask] += expert_out*cof_k

        return out

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


class FourierAmp(nn.Module):
    def __init__(self, dim):
        super(FourierAmp, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(dim, dim, 1,1,0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1,1,0)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_amp = torch.abs(x)
        x_pha = torch.angle(x)
        x_amp = self.process(x_amp)
        out = x_amp*torch.exp(1j*x_pha)
        
        return out


class FourierPha(nn.Module):
    def __init__(self, dim):
        super(FourierPha, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(dim, dim, 1,1,0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1,1,0)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        x_amp = torch.abs(x)
        x_pha = torch.angle(x)
        x_pha = self.process(x_pha)
        out = x_amp*torch.exp(1j*x_pha)
        
        return out

##########################################################################
## Frequency Decoupling Learning Block (DFLB)
class DFLB(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(DFLB, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )

        self.fusion = MoFE(dim)

        self.fourier_pha = FourierPha(dim)
        self.fourier_amp = FourierAmp(dim)

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')

        high_feature, low_feature = self.fft(x) 

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature  = self.channel_cross_h(low_feature, y)

        out = self.fusion(torch.cat([low_feature, high_feature], dim=1))

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)

        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        # h,w = torch.floor(alpha*H),torch.floor(beta*W)

        for i in range(mask.shape[0]):
            # h_ = (h//n * threshold[i,0,:,:]).int()
            # w_ = (w//n * threshold[i,1,:,:]).int()
            # mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

            h_ = torch.floor(threshold[i,0,:,:] * h).int()
            w_ = torch.floor(threshold[i,1,:,:] * w).int()
            mask[i, :, (h-h_)//2:(h+h_)//2, (w-w_)//2:(w+w_)//2] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        # 添加操作 高频只对相位处理
        # ==============================
        high = self.fourier_pha(high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        # 添加操作 低频只对振幅处理
        # ==============================
        low = self.fourier_amp(low)
        # ==============================
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low

class MOENet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [1,1,1,1], 
        bias = True,
        heads = [1,2,4,8],
        decoder = True
    ):

        super(MOENet, self).__init__()

        self.first = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        self.encoder_level1 = nn.Sequential(*[MEB(in_size=dim, out_size=dim) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[MEB(in_size=int(dim*2**1), out_size=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[MEB(in_size=int(dim*2**2), out_size=int(dim*2**2)) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[MEB(in_size=int(dim*2**3), out_size=int(dim*2**3)) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[MEB(in_size=int(dim*2**2), out_size=int(dim*2**2)) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[MEB(in_size=int(dim*2**1), out_size=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 

        self.decoder_level1 = nn.Sequential(*[MEB(in_size=int(dim*2**1), out_size=int(dim*2**1)) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[MEB(in_size=int(dim*2**1), out_size=int(dim*2**1)) for i in range(num_blocks[0])])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder = decoder
        if self.decoder:
            self.fre1 = DFLB(dim*2**3, num_heads=heads[2], bias=bias)
            self.fre2 = DFLB(dim*2**2, num_heads=heads[2], bias=bias)
            self.fre3 = DFLB(dim*2**1, num_heads=heads[2], bias=bias)   

    def forward(self, inp_img):

        inp_enc_level1 = self.first(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        if self.decoder:
            latent = self.fre1(inp_img, latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    model = MOENet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))


