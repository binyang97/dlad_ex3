# from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.task2 import enlarge_box

def compute_inv_matrix(box):
    x = box[0]
    y = box[1]
    z = box[2]
    ry = box[6]

    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    T = [[cos_ry, 0, sin_ry, x],
        [0,       1, 0,      y],
        [-sin_ry, 0, cos_ry, z]]
    T = np.array(T)
    C = T[:3, :3]
    r = T[:3,  3]
    T_inv = np.zeros((3,4))
    T_inv[:,:3] = C.T
    T_inv[:, 3] = np.dot(-C.T, r)

    return T_inv

def voxelization(proposals, xyzs, feats, config):
    # shuffling the points
    #np.random.shuffle(xyzs)
    #voxel_coords = []
    voxel_features = []
    enlarged_proposals = enlarge_box(proposals, config['delta'])
    voxel_size = config['voxel_grid_size']
    
    for (p, proposal) in enumerate(enlarged_proposals):
        h, w, l = proposal[3], proposal[4], proposal[5]
        feat = feats[p]
        xyz_global = xyzs[p]

        if not config['use_ccs']:
            T_inv = compute_inv_matrix(proposal)
            xyz_global_homo = np.hstack([xyz_global, np.ones((len(xyz_global),1))]).T
            xyz_canonical =  np.dot(T_inv, xyz_global_homo).T
            xyz = xyz_canonical[:, :3]
            voxel_coord = np.array([[l*i/voxel_size + l/(voxel_size*2), -h*j/voxel_size-h/(voxel_size*2), w*k/voxel_size + w/(voxel_size*2)] \
                                for i in range(-int(voxel_size/2), int(voxel_size/2)) for j in range(voxel_size) for k in range(-int(voxel_size/2), int(voxel_size/2))])
        else:
            xyz = xyz_global
            voxel_coord = np.array([[l*i/voxel_size + l/(voxel_size*2), -h*j/voxel_size-h/(voxel_size*2) + 1, w*k/voxel_size + w/(voxel_size*2)] \
                                for i in range(-int(voxel_size/2), int(voxel_size/2)) for j in range(voxel_size) for k in range(-int(voxel_size/2), int(voxel_size/2))])
        
        
        xmax = np.max(xyz[:,0]) + l/(voxel_size*2)
        xmin = np.min(xyz[:,0]) - l/(voxel_size*2)
        ymax = np.max(xyz[:,1]) + h/(voxel_size*2)
        ymin = np.min(xyz[:,1]) - h/(voxel_size*2)
        zmax = np.max(xyz[:,2]) + w/(voxel_size*2)
        zmin = np.min(xyz[:,2]) - w/(voxel_size*2)

        voxel_mask = (voxel_coord[:,0]>xmin) & (voxel_coord[:,0]<xmax) & \
                    (voxel_coord[:,1]>ymin) & (voxel_coord[:,1]<ymax)& \
                    (voxel_coord[:,2]>zmin) & (voxel_coord[:,2]<zmax)

        voxel_idx = np.where(voxel_mask)[0]
        

        voxel_coord_filtered = voxel_coord[voxel_idx]

        voxel_feature = np.zeros((config['max_num_voxels'], config['max_num_points_per_voxel'], feat.shape[1]+3))
        count = np.zeros(config['max_num_voxels'], dtype = int)

        dist2voxel = np.array([np.linalg.norm(voxel_coord_filtered - point, axis = 1) for point in xyz]) 

        selected_voxel_coord = voxel_idx[np.argmin(dist2voxel, axis = 1)]
                
        for point_idx, voxel_idx in enumerate(selected_voxel_coord):
            if count[voxel_idx] < config['max_num_points_per_voxel']:
                
                voxel_feature[voxel_idx, count[voxel_idx]] = np.hstack((feat[point_idx], xyz_global[point_idx]))
                count[voxel_idx] += 1
        
        voxel_feature = np.squeeze(np.sum(voxel_feature, axis = 1)) #(Voxel, C)
        for i, c in enumerate(count):
            if c == 0:
                pass
            else:
                voxel_feature[i] = voxel_feature[i] / c

        #voxel_coords.append(voxel_coord)
        voxel_feature = np.array(voxel_feature).flatten().reshape(-1,1)
        voxel_features.append(voxel_feature)

    return np.array(voxel_features)


class Voxelization(nn.Module):
    def __init__(self,config):
        super(Voxelization, self).__init__()
        self.config = config

    def compute_inv_matrix(self, box):
        h = box[3]
        w = box[4]
        l = box[5]
        x = box[0]
        y = box[1]
        z = box[2]
        ry = box[6]

        cos_ry = torch.cos(ry)
        sin_ry = torch.sin(ry)
        T = torch.Tensor([[cos_ry, 0, sin_ry, x],
                        [0,       1, 0,      y],
                        [-sin_ry, 0, cos_ry, z]])
        C = T[:3, :3]
        r = T[:3,  3]
        T_inv = torch.zeros((3,4))
        T_inv[:,:3] = C.t()
        T_inv[:, 3] = torch.matmul(-C.t(), r)

        return T_inv

    def forward(self, x):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        proposals = x['proposal'].contiguous()   
        xyzs, feats = x['xyz_feat'][..., :3].contiguous(), x['xyz_feat'][...,3:].contiguous()   
        voxel_features = []
        enlarged_proposals = enlarge_box(proposals, self.config['delta'])
        voxel_size = self.config['voxel_grid_size']
        for (p, proposal) in enumerate(enlarged_proposals):
            h, w, l = proposal[3], proposal[4], proposal[5]
            feat = feats[p]
            xyz_global = xyzs[p]

            T_inv = self.compute_inv_matrix(proposal)
            xyz_global_homo = torch.hstack((xyz_global, torch.ones((len(xyz_global),1)))).t()
            xyz_canonical =  torch.matmul(T_inv, xyz_global_homo).t()
            xyz = xyz_canonical[:, :3]

            voxel_coord = torch.Tensor([[l*i/voxel_size + l/(voxel_size*2), -h*j/voxel_size-h/(voxel_size*2), w*k/voxel_size + w/(voxel_size*2)] \
                                    for i in range(-int(voxel_size/2), int(voxel_size/2)) for j in range(voxel_size) for k in range(-int(voxel_size/2), int(voxel_size/2))])

            xmax = torch.max(xyz[:,0]) + l/(voxel_size*2)
            xmin = torch.min(xyz[:,0]) - l/(voxel_size*2)
            ymax = torch.max(xyz[:,1]) + h/(voxel_size*2)
            ymin = torch.min(xyz[:,1]) - h/(voxel_size*2)
            zmax = torch.max(xyz[:,2]) + w/(voxel_size*2)
            zmin = torch.min(xyz[:,2]) - w/(voxel_size*2)

            voxel_mask = (voxel_coord[:,0]>xmin) & (voxel_coord[:,0]<xmax) & \
                        (voxel_coord[:,1]>ymin) & (voxel_coord[:,1]<ymax)& \
                        (voxel_coord[:,2]>zmin) & (voxel_coord[:,2]<zmax)

            voxel_idx = torch.where(voxel_mask)[0]
            

            voxel_coord_filtered = voxel_coord[voxel_idx]

            voxel_feature = torch.zeros((self.config['max_num_voxels'], self.config['max_num_points_per_voxel'], feat.shape[1]+3))
            count = torch.zeros(self.config['max_num_voxels'], dtype = int)

            dist2voxel = torch.stack([torch.norm(voxel_coord_filtered - point, dim = 1) for point in xyz]) 

            selected_voxel_coord = voxel_idx[torch.argmin(dist2voxel, dim = 1)]
                    
            for point_idx, voxel_idx in enumerate(selected_voxel_coord):
                if count[voxel_idx] < self.config['max_num_points_per_voxel']:
                    
                    voxel_feature[voxel_idx, count[voxel_idx]] = torch.cat((feat[point_idx], xyz_global[point_idx]))
                    count[voxel_idx] += 1
            

            #voxel_coords.append(voxel_coord)
            voxel_features.append(voxel_feature)

        return torch.stack((voxel_features))



# Fully Connected Network
class FCN(nn.Module):
    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
       # print(x.shape)
        d, kk, t, _ = x.shape
        #x = self.linear(x.view(kk*t,-1))
        x = self.linear(x)
        x = x.view(d*kk*t, -1)
        x = F.relu(self.bn(x))
        return x.view(d,kk,t,-1)

# Voxel Feature Encoding layer
class VFE(nn.Module):
    def __init__(self,cin,cout,T):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin,self.units)
        self.T = T

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        #locally aggregated feature
        laf = torch.max(pwf,2)[0].unsqueeze(2).repeat(1, 1,self.T,1)
        # point-wise concat feature
        pwcf = torch.cat((pwf,laf),dim=3)
        # apply mask
        mask = mask.unsqueeze(3).repeat(1, 1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(nn.Module):
    def __init__(self, config):
        super(SVFE, self).__init__()
        self.__dict__.update(config)

        channel_in = self.num_point_features
        self.vfes = nn.ModuleList()
        for k in range(len(self.vfe_fc)):
            self.vfes.append(VFE(channel_in, self.vfe_fc[k], self.max_num_points_per_voxel))
            channel_in = self.vfe_fc[k]

        self.fcn = FCN(channel_in,channel_in)

    def forward(self, x):
        mask = torch.ne(torch.max(x,3)[0], 0)

        for layer in self.vfes:
            x = layer(x, mask)

        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x,2)[0].reshape(x.size(0), -1)

        assert x.shape == (x.size(0), self.max_num_voxels * self.num_point_features)
        return x

class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:
        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError

class MeanVFE(nn.Module):
    def __init__(self, config):
        super(MeanVFE, self).__init__()
        self.__dict__.update(config)
        #self.num_point_features = num_point_features

    #def get_output_feature_dim(self):
    #    return self.num_point_features

    def forward(self, x):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:
        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = x, self.num_point_features
        points_mean = voxel_features[:, :, :, :].mean(dim=2, keepdim=False)
        #normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        #points_mean = points_mean / normalizer
        voxel_features = points_mean.contiguous()
        batch_size, _, _ =  voxel_features.shape
        voxel_features = voxel_features.reshape(batch_size, -1, 1)

        return voxel_features

# tv = None
# try:
#     import cumm.tensorview as tv
# except:
#     pass

# try:
#     import spconv.pytorch as spconv
# except:
#     import spconv as spconv

# def replace_feature(out, new_features):
#     if "replace_feature" in out.__dir__():
#         # spconv 2.x behaviour
#         return out.replace_feature(new_features)
#     else:
#         out.features = new_features
#         return out

# def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
#                    conv_type='subm', norm_fn=None):

#     if conv_type == 'subm':
#         conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
#     elif conv_type == 'spconv':
#         conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                    bias=False, indice_key=indice_key)
#     elif conv_type == 'inverseconv':
#         conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
#     else:
#         raise NotImplementedError

#     m = spconv.SparseSequential(
#         conv,
#         norm_fn(out_channels),
#         nn.ReLU(),
#     )

#     return m


# class SparseBasicBlock(spconv.SparseModule):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
#         super(SparseBasicBlock, self).__init__()

#         assert norm_fn is not None
#         bias = norm_fn is not None
#         self.conv1 = spconv.SubMConv3d(
#             inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn1 = norm_fn(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = spconv.SubMConv3d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn2 = norm_fn(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = replace_feature(out, self.bn1(out.features))
#         out = replace_feature(out, self.relu(out.features))

#         out = self.conv2(out)
#         out = replace_feature(out, self.bn2(out.features))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = replace_feature(out, out.features + identity.features)
#         out = replace_feature(out, self.relu(out.features))

#         return out


# class VoxelBackBone(nn.Module):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

#         self.sparse_shape = grid_size[::-1] + [1, 0, 0]

#         self.conv_input = spconv.SparseSequential(
#             spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#         block = post_act_block

#         self.conv1 = spconv.SparseSequential(
#             block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
#         )

#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#         )

#         self.conv3 = spconv.SparseSequential(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#         )

#         self.conv4 = spconv.SparseSequential(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#         )

#         last_pad = 0
#         last_pad = self.model_cfg.get('last_pad', last_pad)
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 bias=False, indice_key='spconv_down2'),
#             norm_fn(128),
#             nn.ReLU(),
#         )
#         self.num_point_features = 128
#         self.backbone_channels = {
#             'x_conv1': 16,
#             'x_conv2': 32,
#             'x_conv3': 64,
#             'x_conv4': 64
#         }



#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (config['max_num_voxels'], C)
#                 voxel_coords: (config['max_num_voxels'], 4), [batch_idx, z_idx, y_idx, x_idx]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         """
#         voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=voxel_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )

#         x = self.conv_input(input_sp_tensor)

#         x_conv1 = self.conv1(x)
#         x_conv2 = self.conv2(x_conv1)
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)

#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)

#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#         batch_dict.update({
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#             }
#         })
#         batch_dict.update({
#             'multi_scale_3d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#             }
#         })

#         return batch_dict

# class VoxelGeneratorWrapper():
#     def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
#         try:
#             from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
#             self.spconv_ver = 1
#         except:
#             try:
#                 from spconv.utils import VoxelGenerator
#                 self.spconv_ver = 1
#             except:
#                 from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
#                 self.spconv_ver = 2

#         if self.spconv_ver == 1:
#             self._voxel_generator = VoxelGenerator(
#                 voxel_size=vsize_xyz,
#                 point_cloud_range=coors_range_xyz,
#                 max_num_points=max_num_points_per_voxel,
#                 max_voxels=max_num_voxels
#             )
#         else:
#             self._voxel_generator = VoxelGenerator(
#                 vsize_xyz=vsize_xyz,
#                 coors_range_xyz=coors_range_xyz,
#                 num_point_features=num_point_features,
#                 max_num_points_per_voxel=max_num_points_per_voxel,
#                 max_num_voxels=max_num_voxels
#             )

#     def generate(self, points):
#         if self.spconv_ver == 1:
#             voxel_output = self._voxel_generator.generate(points)
#             if isinstance(voxel_output, dict):
#                 voxels, coordinates, num_points = \
#                     voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
#             else:
#                 voxels, coordinates, num_points = voxel_output
#         else:
#             assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
#             voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
#             tv_voxels, tv_coordinates, tv_num_points = voxel_output
#             # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
#             voxels = tv_voxels.numpy()
#             coordinates = tv_coordinates.numpy()
#             num_points = tv_num_points.numpy()
#         return voxels, coordinates, num_points

# def points_to_voxels(proposals, xyzs, feats, config):
#     voxels = []
#     voxel_coords = []

#     points = np.concatenate([feats, xyzs], axis=-1)
#     enlarged_proposals = enlarge_box(proposals, config['delta'])
#     for (p, proposal) in enumerate(enlarged_proposals):
#         h, w, l = proposal[3], proposal[4], proposal[5]
#         point_cloud_range = np.array([-l/2, -h, -w/2, l/2, 0, w/2])
#         grid_size = np.array(config['grid_size'])
#         voxel_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / grid_size

#         voxel_generator = VoxelGeneratorWrapper(
#             vsize_xyz=voxel_size,
#             coors_range_xyz=point_cloud_range,
#             num_point_features=points.shape[-1],
#             max_num_points_per_voxel=config['max_num_points_per_voxel'],
#             max_num_voxels=config['max_num_voxels'],
#         )

#         voxel_output = voxel_generator.generate(points)
#         voxel, coordinate, num_point = voxel_output
#         box_id = p*np.ones((coordinate.shape[0], 1)).astype(int)
#         coordinate = np.concatenate([box_id, coordinate], axis=-1)

#         voxels.append(voxel) #[#non_empty, 35, #feature]
#         voxel_coords.append(coordinate)  #[#non_empty, 3]

#     voxels = np.concatenate(voxels, axis=0).reshape(voxels.shape[0],-1) #[#proposals * #non_empty, 35 * #feature]
#     voxel_coords = np.concatenate(voxel_coords, axis=0)  #[#proposals * #non_empty, 3]

#     return voxels, voxel_coords