from unittest.main import main
from xml.dom import NotFoundErr
from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math

class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_kps, num_channels, 
                 scale_factor=0.25, bg = False, fg = False, multi_mask = True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        if bg and fg:
            self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_tps + 2) + num_tps * num_kps + 2),
                                   max_features=max_features, num_blocks=num_blocks)
        else:
            self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_tps + 1) + num_tps * num_kps + 1),
                                   max_features=max_features, num_blocks=num_blocks)
            # 包含了只有一个变换和两个变换都没有，补充了一个通道的情况，
            # 训练的时候都是带着变换推光流的，但是去掉补充一个新的之后能有作用么？对于不变的背景倒是有用。


        hourglass_output_size = self.hourglass.out_channels
        if bg and fg:
            self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 2, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))
            
        if multi_mask:
            up = []
            self.up_nums = int(math.log(1/scale_factor, 2)) # 2
            self.occlusion_num = 4
            
            channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up) # 2 levels

            channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1]//(2**(i+1)))
            occlusion = []
            
            for i in range(self.occlusion_num-1):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            occlusion.append(nn.Conv2d(channel[self.occlusion_num-1], 2, kernel_size=(7, 7), padding=(3, 3)))
            # 此处有修改
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.num_kps = num_kps
        self.bg = bg
        self.fg = fg
        self.kp_variance = kp_variance

        
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance) ## bs, KN, w, h
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance) ## bs, KN, w, h
        heatmap = gaussian_driving - gaussian_source ## bs, KN, w, h

        if self.bg and self.fg:
            zeros = torch.zeros(heatmap.shape[0], 2, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        else:
            zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        
        heatmap = torch.cat([zeros, heatmap], dim=1) ## bs, KN+1(+2), w, h

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param, fg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        kp_1 = kp_1.view(bs, -1, self.num_kps, 2)
        kp_2 = kp_2.view(bs, -1, self.num_kps, 2)
        trans = TPS(mode = 'kp', bs = bs, kp_1 = kp_1, kp_2 = kp_2)
        driving_to_source = trans.transform_frame(source_image) # bs K h w 2

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1) # bs 1 h w 2

        # affine background transformation
        if not (bg_param is None):
            identity_grid_bg = to_homogeneous(identity_grid)
            identity_grid_bg = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid_bg.unsqueeze(-1)).squeeze(-1)
            identity_grid_bg = from_homogeneous(identity_grid_bg) # bs 1 h w 2
        
        # perspective foreground transformation
        if not (fg_param is None):
            identity_grid_fg = to_homogeneous(identity_grid)
            identity_grid_fg = torch.matmul(fg_param.view(bs, 1, 1, 1, 3, 3), identity_grid_fg.unsqueeze(-1)).squeeze(-1)
            identity_grid_fg = from_homogeneous(identity_grid_fg) # bs 1 h w 2

        # transformations = torch.cat([identity_grid_bg, identity_grid_fg, driving_to_source], dim=1) # bs K+2 h w 2
        transformations = driving_to_source # bs K h w 2
        if self.fg:
            if not (fg_param is None):
                transformations = torch.cat([identity_grid_fg, transformations], dim=1)
            else:
                transformations = torch.cat([identity_grid, transformations], dim=1)
                # 这里是在测试的时候满足模型的size要求
        
        if self.bg:
            if not (bg_param is None):
                transformations = torch.cat([identity_grid_bg, transformations], dim=1)
            else:
                transformations = torch.cat([identity_grid, transformations], dim=1)
                # 这里是在测试的时候满足模型的size要求

        return transformations

    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        K = transformations.size(1)
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, K, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * K, -1, h, w)
        transformations = transformations.view((bs * K, h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, K, -1, h, w))
        return deformed # bs K+2 3 h w

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition  

    def forward(self, source_image, kp_driving, kp_source, bg_param = None, fg_param = None, dropout_flag=False, dropout_p = 0):
        if self.scale_factor != 1:
            source_image = self.down(source_image) ## /4 downsample

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source) # bs KN+2(+1) h w
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param, fg_param) # bs K+2(+1) h w 2

        deformed_source = self.create_deformed_source_image(source_image, transformations) # bs K+2(+1) 3 h w 输入图像使用每一种变形搞一下
        out_dict['deformed_source'] = deformed_source
        
        deformed_source = deformed_source.view(bs,-1,h,w)
        input = torch.cat([heatmap_representation, deformed_source], dim=1) # 形变之后的图像和关键点一起输入
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input, mode = 1)

        contribution_maps = self.maps(prediction[-1]) ## bs k+2 h w
        if(dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the K+2 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2) ## bs k+2 1 h w
        transformations = transformations.permute(0, 1, 4, 2, 3) # bs K+2 2 h w
        deformation = (transformations * contribution_maps).sum(dim=1) # bs 2 h w 使用contribution map给transformation加了权
        deformation = deformation.permute(0, 2, 3, 1) # bs h w 2

        out_dict['deformation'] = deformation # Optical Flow 可以用这个算出光流，但是并不直接是光流

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num-self.up_nums):
                occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums-self.occlusion_num+i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))

        out_dict['attention_map'] = [occlusion_map[-1][:,1:]]
        occlusion_map[-1] = occlusion_map[-1][:,:1]
        out_dict['occlusion_map'] = occlusion_map # Multi-resolution Occlusion Masks 
        # 32x32x1 64x64x1 128x128x1 256x256x1
        return out_dict

if __name__=='__main__':
    model = DenseMotionNetwork(64, 5, 1024, 10, 8, 3, scale_factor=0.25, bg = True, multi_mask = True, kp_variance=0.01)
    print(model)
