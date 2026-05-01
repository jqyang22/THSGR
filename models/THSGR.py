import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.transformer import ViT
from einops import rearrange, repeat
class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=20):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x
    def forward_construct_dynamic_graph(self, x):
        m_batchsize, C, class_num = x.size()
        proj_query = x
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        dynamic_adj = torch.bmm(proj_key, proj_query)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        return dynamic_adj, out
    def forward_dynamic_gcn(self, x, dynamic_adj):
        weight = self.dynamic_weight(x)
        support = torch.mul(x, weight)
        x = torch.matmul(support, dynamic_adj)
        x = self.relu(x)
        return x
    def forward(self, x, sds=[0, 1, 0]):
        static, dynamic, static_dynamic = sds
        if static:
            out_static = self.forward_static_gcn(x)
        if dynamic:
            dynamic_adj, out = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x, dynamic_adj)
        if static_dynamic:
            out_static = self.forward_static_gcn(x)
            x = x + out_static
            dynamic_adj, out = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x, out
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask
    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value,
                                       stop=stop_value, num=int(nr_steps))
    def forward(self, x):
        return self.dropblock(x)
    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1
class THSGR(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes, patch_size, drop_prob=0.1, block_size=3):
        super(THSGR, self).__init__()
        self.input_channels = input_channels
        self.num_node = num_nodes
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                                         start_value=0.,
                                         stop_value=drop_prob,
                                         nr_steps=5e3)
        self.conv1 = nn.Conv2d(self.input_channels, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_3D = nn.Conv3d(1, 8, (7, 3, 3))
        self.bn1_3D = nn.BatchNorm3d(8)
        self.conv2_3D = nn.Conv3d(8, 16, (5, 3, 3))
        self.bn2_3D = nn.BatchNorm3d(16)
        self.conv3_3D = nn.Conv3d(16, 32, (3, 3, 3))
        self.bn3_3D = nn.BatchNorm3d(32)
        self.conv4_2D = nn.Conv2d(32*20, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.features_size = self._get_final_flattened_size()
        self.fc_sam = nn.Conv2d(64, self.num_node, (1, 1), bias=False)
        self.conv_transform = nn.Conv2d(64, 64, (1, 1))
        self.gcn = DynamicGraphConvolution(64, 64, num_nodes=self.num_node)
        self.gcn_lidar = DynamicGraphConvolution(64, 64, num_nodes=self.num_node)
        self.transformer = ViT(
            image_size=patch_size,
            near_band=1,
            num_patches=64,
            num_classes=num_classes,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            mode='ViT'
        )
        self.fc4 = nn.Linear(64, 256)
        self.fc5 = nn.Linear(256, self.num_classes)
        self.fc6 = nn.Linear(225, 64)
        self.fc7 = nn.Linear(64 * num_classes, 64)
        self.fc10 = nn.Linear(64 * num_classes, 64 * 64)
        self.fc8 = nn.Linear(64 * 2, 256)
        self.fc9 = nn.Linear(num_classes, 64)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(self.features_size, 256)
        self.drop1 = nn.Dropout(0.5)
        self.bn_f1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_f2 = nn.BatchNorm1d(
            256)
        self.fc3 = nn.Linear(256, self.num_classes)
        self.conv1_LiDAR = nn.Conv2d(1, 32, 3)
        self.conv1_LiDAR_2rasters = nn.Conv2d(4, 32, 3)
        self.bn1_LiDAR = nn.BatchNorm2d(32)
        self.conv2_LiDAR = nn.Conv2d(32, 64, 3)
        self.bn2_LiDAR = nn.BatchNorm2d(64)
        self.lidarConv1 = nn.Sequential(
                        nn.Conv2d(1,64,3),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )
        self.lidarConv2 = nn.Sequential(
                        nn.Conv2d(64,64,3),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )
        self.lidarConv3 = nn.Sequential(
                        nn.Conv2d(64,64,3),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )
        self.lidarfc1 = nn.Linear(patch_size*patch_size, 64)
        self.lidarfc2 = nn.Linear(683, 64)
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x_pool = self.pool1(x)
            x = self.conv3(x_pool)
            x = self.conv4(x)
            x = self.avgpool(x)
            _, c, w, h = x.size()
            x = self.conv5(x_pool)
            x = self.conv6(x)
            _, c2, w2, h2 = x.size()
        return c * w * h + c2 * w2 * h2 + 64*self.num_node
    def forward_sam(self, x):
        mask = self.fc_sam(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x
    def forward_sam_lidar(self, x):
        mask = self.fc_sam(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x
    def forward(self, x, x_LiDAR, w):
        print('x', x.shape)
        print('x_LiDAR', x_LiDAR.shape)
        self.dropblock.step()
        x1 = F.leaky_relu(self.conv1_3D(x))
        x1 = self.bn1_3D(x1)
        x2 = F.leaky_relu(self.conv2_3D(x1))
        x2 = self.bn2_3D(x2)
        x3 = F.leaky_relu(self.conv3_3D(x2))
        x3 = self.bn3_3D(x3)
        x3 = torch.reshape(x3, (x3.shape[0], x3.shape[1]*x3.shape[2], x3.shape[3], x3.shape[4]))
        x4 = F.leaky_relu(self.conv4_2D(x3))
        x7 = self.forward_sam(x4)
        gcn, proj_out = self.gcn(x7)
        gcn = gcn.view(-1, gcn.size(1) * gcn.size(2))
        gcn = self.fc7(gcn)
        self.dropblock.step()
        if len(x_LiDAR.shape) == 5:
            x_LiDAR = x_LiDAR.squeeze(1)
            x_LiDAR = x_LiDAR.permute(0, 3, 1, 2)
            x_LiDAR = torch.mean(x_LiDAR, dim=1)
            x_LiDAR = torch.unsqueeze(x_LiDAR, 1)
            x1_LiDAR = F.leaky_relu(self.conv1_LiDAR(x_LiDAR))
        else:
            x1_LiDAR = F.leaky_relu(self.conv1_LiDAR(x_LiDAR))
        x1_LiDAR = self.bn1_LiDAR(x1_LiDAR)
        x2_LiDAR = F.leaky_relu(self.conv2(x1_LiDAR))
        x2_LiDAR = self.bn2_LiDAR(x2_LiDAR)
        x7_LiDAR = self.forward_sam(x2_LiDAR)
        _, proj_out_LiDAR = self.gcn(x7_LiDAR)
        proj_out_LiDAR = self.fc9(proj_out_LiDAR)
        proj_out = torch.reshape(self.fc10(proj_out.view(-1, proj_out.size(1) * proj_out.size(2))), (proj_out_LiDAR.shape[0], proj_out_LiDAR.shape[1], proj_out_LiDAR.shape[2]))
        tr = self.transformer(proj_out, proj_out_LiDAR)
        x = F.leaky_relu(self.fc4((1-w) * tr + w * gcn))
        x = self.bn_f1(x)
        x = self.drop1(x)
        x = self.fc5(x)
        return x
