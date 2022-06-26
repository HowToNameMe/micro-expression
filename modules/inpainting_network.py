from numpy import source
import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
# from modules.dense_motion import DenseMotionNetwork

class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features//2]
        out_features = [max_features//2, max_features//4, max_features//8]
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, source_mask, dense_motion):

        # Shared Encoder source image features 
        out = self.first(source_image)
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        # masked image
        masked_image = source_image
            
        # Shared Encoder masked image features
        out_masked = self.first(masked_image)
        encoder_map_masked = [out_masked]
        for i in range(len(self.down_blocks)):
            out_masked = self.down_blocks[i](out_masked)
            encoder_map_masked.append(out_masked)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']

        # occlusion_bg
        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        attention_map = dense_motion['attention_map']
        output_dict['attention_map'] = attention_map

        deformation = dense_motion['deformation']
        out_masked_ij = self.deform_input(out_masked.detach(), deformation) # 这一步是为了记录, deformation 无法参与此处的梯度
        out_masked = self.deform_input(out_masked, deformation)

        out_masked_ij = self.occlude_input(out_masked_ij, occlusion_map[0].detach())
        out_masked = self.occlude_input(out_masked, occlusion_map[0])

        warped_encoder_maps = []
        warped_encoder_maps.append(out_masked_ij)

        for i in range(self.num_down_blocks):
            
            out_masked = self.resblock[2*i](out_masked)
            out_masked = self.resblock[2*i+1](out_masked)
            out_masked = self.up_blocks[i](out_masked)
            
            encode_masked_i = encoder_map_masked[-(i+2)]
            encode_masked_ij = self.deform_input(encode_masked_i.detach(), deformation)
            encode_masked_i = self.deform_input(encode_masked_i, deformation)
            
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_masked_ij = self.occlude_input(encode_masked_ij, occlusion_map[occlusion_ind].detach())
            encode_masked_i = self.occlude_input(encode_masked_i, occlusion_map[occlusion_ind])

            warped_encoder_maps.append(encode_masked_ij)

            if(i==self.num_down_blocks-1):
                break

            out_masked = torch.cat([out_masked, encode_masked_i], 1)

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps


        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear', align_corners=True)
            
        out_masked = out_masked * (1 - occlusion_last) + encode_masked_i
        out_masked = self.final(out_masked)
        out_masked = torch.sigmoid(out_masked)

        out_masked = out_masked * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out_masked

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2-i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map

