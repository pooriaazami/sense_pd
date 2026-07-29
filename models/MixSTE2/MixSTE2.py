from functools import partial

import torch
import torch.nn as nn

from .modules import *


class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 512     #### output dimension is num_joints * 3
        self.is_train=is_train

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )



    def STE_forward(self, x, t):

        if self.is_train:
            b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = x.reshape(-1, x.shape[2], x.shape[3]) # x = rearrange(x, 'b f n c  -> (b f) n c', )
            ### now x is [batch_size, receptive frames, joint_num, 2 channels]
            # x = self.Spatial_patch_to_embedding(x)
            # x = rearrange(x, 'bnew c n  -> bnew n c', )
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1,f,n,1)
            time_embed = time_embed.reshape(-1, time_embed.shape[2], time_embed.shape[3]) # time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            b, h, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = x.reshape(-1, x.shape[3], x.shape[4]) # x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            # x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, h, f, n, 1)
            time_embed = time_embed.reshape(-1, time_embed.shape[3], time_embed.shape[4]) # time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        x = x.reshape(-1, f, x.shape[1], x.shape[2]) \
            .permute(0, 2, 1, 3) \
            .reshape(-1, f, x.shape[2]) # x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = x.reshape(-1, x.shape[2], x.shape[3]) # x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = x.reshape(-1, f, x.shape[1], x.shape[2]).permute(0, 2, 1, 3).reshape(-1, f, x.shape[2]) # x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = x.reshape(b, n, f, cw).permute(0, 2, 1, 3).contiguous() # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def forward(self, x_3d, t, x_cond):
        x_cond = x_cond + self.Temporal_pos_embed[:,0:1,:].unsqueeze(dim=2)
        if self.is_train:
            b,f,n,c, = x_3d.shape
        else:
            b, h, f, n, c = x_3d.shape
            x_cond = x_cond.unsqueeze(dim=1)
        
        x = x_3d + x_cond
        x = self.STE_forward(x_3d, t)

        x = self.TTE_foward(x)

        x = x.reshape(-1, n, x.shape[1], x.shape[2]).permute(0, 2, 1, 3) # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x