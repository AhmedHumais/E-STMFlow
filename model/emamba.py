from argparse import Namespace
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from mamba_ssm import Mamba
# Mamba2=None
# from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm import Mamba2
from .ssm_block import Block
from .s4.s4 import S4Block

from calflops import calculate_flops

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        # self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class UpNx(nn.Module):
    def __init__(self, in_dim, out_dim=1, N=2):
        super().__init__()
        self.out_dim = out_dim
        self.N = N
        self.transfrom = nn.Conv3d(in_dim, self.out_dim*(self.N**2), kernel_size=1)
        
    def forward(self, x):
        x = self.transfrom(x)
        B, C, T, H, W = x.shape 
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = x.reshape(B * T,self.out_dim, self.N, self.N, H, W)    ### check if reshape is required here
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, T, self.out_dim, self.N*H, self.N*W)
        
        return x.permute(0, 2, 1, 3, 4)               #to return (B, C, T, H, W) tensors

class MambaX2(nn.Module):
    def __init__(self, dim, num_layers=2, mamba_type=1):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(Block(dim, Mamba if mamba_type==1 else Mamba2))

    def forward(self, x, residual=None):
        for block in self.blocks:
            if residual is not None:
                x, residual = block(x, residual)
            else:
                x, residual = block(x)
        return x, residual

class SSM_Block(nn.Module):
    def __init__(self, dim, num_layers=2, ssm_type="s4"):
        super().__init__()
        if ssm_type == "s5":
            from s5 import S5Block
        elif ssm_type == "s4d":
            from .s4.s4d import S4D
        self.ssm_type = ssm_type
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if ssm_type == "s5":
                self.blocks.append(S5Block(dim, 64, False))
            elif ssm_type == "s4d":
                self.blocks.append(S4D(dim, transposed=False))
            else:
                self.blocks.append(S4Block(dim, transposed=False))                

    def forward(self, x, residual=None):
        if residual is not None:
            x = x+residual
        for block in self.blocks:
            residual = x
            if self.ssm_type == "s5":
                x = block(x)
            else:
                x, _ = block(x)
        return x, residual

class Transformer(nn.Module):
    def __init__(self, dim, num_layers=2):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.TransformerEncoderLayer(d_model=dim, nhead=2, batch_first=True, norm_first=True))
        
    def forward(self, x, residual=None):
        if residual is not None:
            x = x+residual
        for block in self.blocks:
            residual = x
            x = block(x)
        return x, residual
         
class MambaLayer(nn.Module):
    def __init__(
            self, 
            patch_size,
            img_size=None, 
            n_bins=24,
            t_win_sz=1, 
            embed_dim=32,
            dim=1,
            drop_path_rate=0.1,
            device=None,
            dtype=None,
            drop_rate =0.0,
            use_checkpoint=False,
            use_transformer=False,
            num_mixers =2, 
            v_mamba = 1,
            p_enc = False,
            ssm_type=None
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()

        assert t_win_sz <= n_bins

        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        patch_size = to_2tuple(patch_size)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, 
            kernel_size=t_win_sz,
            in_chans=dim, embed_dim=embed_dim
        )

        self.en_pos_enc = p_enc
        if self.en_pos_enc:
            assert img_size is not None
            num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if t_win_sz == n_bins:
            self.temporal_pos_embedding = None
        else:
            self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, n_bins // t_win_sz, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # self.mamba = MambaX2(embed_dim, num_layers=num_mixers, mamba_type=v_mamba) if not use_transformer else Transformer(embed_dim, num_layers=num_mixers)
        if ssm_type is not None:
            print(f"Using SSM of type {ssm_type} ...")
            self.mamba = SSM_Block(embed_dim, num_mixers, ssm_type)
        elif use_transformer:
            print("Using Transformer ...")
            self.mamba = Transformer(embed_dim, num_layers=num_mixers)
        else:
            print(f"Using Mamba v{v_mamba} ...")
            self.mamba = MambaX2(embed_dim, num_layers=num_mixers, mamba_type=v_mamba)
                     
        # output head
        self.norm_f = nn.LayerNorm(embed_dim, **factory_kwargs)

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     return {
    #         i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    #         for i, layer in enumerate(self.layers)
    #     }


    def forward(self, x:torch.Tensor, residual:Optional[torch.Tensor]=None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        if self.en_pos_enc:
            x = x + self.pos_embed

        # temporal pos
        if self.temporal_pos_embedding is not None:
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            x = x + self.temporal_pos_embedding
            x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        else:
            x = rearrange(x, '(b t) n m -> b (t n) m', b=B, t=T)
        
        if self.training:    
            x = self.pos_drop(x)

        # mamba impl


        if residual is None:
            hidden_states, residual = self.mamba(x)
        else:
            hidden_states, residual = self.mamba(x, residual)

        residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        y = rearrange(hidden_states, 'b (t n) m -> b m t n', b=B, t=T)
        y = y.reshape(B, C, T, H, W)

        # return y, residual
        return y

def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    in_args = Namespace(H= 480, 
                     W = 640)

    in_args.n_bins = 32
    in_args.e_dims = [32, 64, 256, 512]
    # in_args.e_dims = [256, 256, 256, 512]
    in_args.dims = [1, 32, 64, 128]
    in_args.T_dims = [32, 8, 4, 1]
    in_args.patch_sizes = [32, 8, 4, 1]
    in_args.up_sizes = [16, 4, 2]    
    in_args.pos_enc = False
    in_args.mamba_v = 1
    in_args.use_trans = False
    in_args.ssm = None
    in_args.n_mixers = 2
    
    return in_args

class EMambFlow(nn.Module):
    def __init__(self, args = None):
        # args:
        super(EMambFlow, self).__init__()
        # args = get_args()
        # self.args = args        
        # H, W = 480, 640
        # # H, W = 288,384        
        # n_bins = 32
        # e_dims = [32, 64, 256, 512]
        # # dims = [1, 16, 32, 128]
        # dims = [1, 32, 64, 128]
        # T_dims = [32, 8, 4, 1]
        # patch_sizes = [32, 8, 4, 1]
        # up_sizes = [16, 4, 2]        
        if args is None:
            args = get_args()

        H, W = args.H, args.W
        n_bins = args.n_bins
        e_dims = args.e_dims
        dims = args.dims
        T_dims = args.T_dims
        patch_sizes = args.patch_sizes
        up_sizes = args.up_sizes             
        use_trans = args.use_trans
        mamba_version = args.mamba_v
        n_mixers = args.n_mixers
        
        assert T_dims[-1] == 1

        # mamba blocks
        out_sz = (H, W)
        self.mm1 = MambaLayer(img_size=out_sz, patch_size=patch_sizes[0], n_bins=n_bins, t_win_sz=n_bins // T_dims[0], dim= dims[0], embed_dim=e_dims[0], 
                              drop_rate=0.0, drop_path_rate=0.0, num_mixers=n_mixers, v_mamba=mamba_version, use_transformer=use_trans, p_enc = args.pos_enc, ssm_type=args.ssm)
        self.up1 = UpNx(in_dim=e_dims[0], out_dim= dims[1], N=up_sizes[0])

        out_sz = ( (out_sz[0] // patch_sizes[0]) * up_sizes[0], (out_sz[1] // patch_sizes[0]) * up_sizes[0] ) 
        self.mm2 = MambaLayer(img_size=out_sz, patch_size=patch_sizes[1], n_bins=T_dims[0], t_win_sz=T_dims[0] // T_dims[1], dim=dims[1], embed_dim=e_dims[1],
                              drop_rate=0.0, drop_path_rate=0.0, num_mixers=n_mixers, v_mamba=mamba_version, use_transformer=use_trans, p_enc = args.pos_enc, ssm_type=args.ssm)
        self.up2 = UpNx(in_dim=e_dims[1], out_dim= dims[2], N=up_sizes[1])

        out_sz = ( (out_sz[0] // patch_sizes[1]) * up_sizes[1], (out_sz[1] // patch_sizes[1]) * up_sizes[1] ) 
        self.mm3 = MambaLayer(img_size=out_sz, patch_size=patch_sizes[2], n_bins=T_dims[1], t_win_sz=T_dims[1] // T_dims[2], dim=dims[2], embed_dim=e_dims[2],
                              drop_rate=0.0, drop_path_rate=0.0, num_mixers=n_mixers, v_mamba=mamba_version, use_transformer=use_trans, p_enc = args.pos_enc, ssm_type=args.ssm)
        self.up3 = UpNx(in_dim=e_dims[2], out_dim= dims[3], N=up_sizes[2])

        out_sz = ( (out_sz[0] // patch_sizes[2]) * up_sizes[2], (out_sz[1] // patch_sizes[2]) * up_sizes[2] ) 
        self.mm4 = MambaLayer(img_size=out_sz, patch_size=patch_sizes[3], n_bins=T_dims[2], t_win_sz=T_dims[2] // T_dims[3], dim=dims[3], embed_dim=e_dims[3],
                              drop_rate=0.0, drop_path_rate=0.0, num_mixers=n_mixers, v_mamba=mamba_version, use_transformer=use_trans, p_enc = args.pos_enc, ssm_type=args.ssm)

        # self.mm4 = MambaLayer(img_size=out_sz, patch_size=patch_sizes[3], n_bins=T_dims[2], t_win_sz=T_dims[2] // T_dims[3], dim=dims[3], embed_dim=e_dims[3],
        #                       drop_rate=0.0, drop_path_rate=0.0, num_mixers=n_mixers, v_mamba=mamba_version, use_transformer=True, p_enc = True)

        out_sz = ( (out_sz[0] // patch_sizes[3]), (out_sz[1] // patch_sizes[3])) 


        self.mask_head = nn.Sequential(
            nn.Conv2d(e_dims[-1], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.flow_head = nn.Sequential(
            nn.Conv2d(e_dims[-1], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 3, padding=1))

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, x: torch.Tensor):
        
        x = x.unsqueeze(1).contiguous()  # [B, C=1, T=32, H=480, W=640]
        
        x = self.mm1(x)
        x = self.up1(x)

        x = self.mm2(x)
        x = self.up2(x)

        x = self.mm3(x)
        x = self.up3(x)

        x = self.mm4(x)

        x = x.squeeze(2).contiguous()  # remove temporal dimension [B, C=9*8*8+2, T=1, H=60, W=80] --> [B, C=9*8*8+2, H=60, W=80]
        flow = self.flow_head(x)   
        up_mask = 0.5*self.mask_head(x)              
        flow_up = self.upsample_flow(flow, up_mask)

        return flow_up, flow


if __name__ == '__main__':
    # from thop import profile, clever_format

    in_tensor = torch.randn(1, 32, 480, 640)

    m1, _ = torch.cuda.mem_get_info()

    net = EMambFlow()
    net = net.to('cuda')
    net.eval()
    inp = in_tensor.float().to('cuda')
    
    in_shape = (1, 32, 480, 640)
    # start = time.time()
    # for i in range(100):
    #     print(torch.cuda.mem_get_info())
    #     functional.reset_net(net)
    # macs, params = profile(net, inputs=(inp, ))
    # macs, params = clever_format([macs, params], "%.3f") 
    flops, macs, params = calculate_flops(model=net, input_shape=in_shape)

    m2, _ = torch.cuda.mem_get_info()

    print((m1-m2)*1e-6)
    
    # print('MACS : {}'.format(macs))
    # print('Params : {}'.format(params))
    # trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('Trainable parameters: {}'.format(trainable_params))
    
    import time
    # times = torch.zeros(1000)
    net.eval()
    for i in range(1001):
        with torch.no_grad():
            _ = net(inp)
        if i==1:
            # times[i-1] = time.time()
            start_time = time.time()        
    print((time.time() - start_time)/1000)        
    # print((torch.diff(times)).mean().item())
    # print(torch.cuda.mem_get_info())
    # functional.reset_net(net)
    # print(torch.cuda.mem_get_info())

    # out_tensor = net(inp)
    # print(torch.cuda.mem_get_info())

    # print(time.time() - start)
    
    # print(out_tensor.shape)