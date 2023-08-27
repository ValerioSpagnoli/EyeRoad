import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.weight_init import init_weights

import numpy as np
import warnings


class BaseComponent(nn.Module):
    def __init__(self, from_layer, to_layer, component):
        super(BaseComponent, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer
        self.component = component

    def forward(self, x):
        return self.component(x)


class FeatureExtractorComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(FeatureExtractorComponent, self).__init__(
            from_layer, 
            to_layer, 
            nn.Sequential(
                GResNet(layers=arch['encoder']['backbone']['layers']), 
                CollectBlock(from_layer=arch['collect']['from_layer']))
        )


class RectificatorComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(RectificatorComponent, self).__init__(
            from_layer, 
            to_layer, 
            TPS_STN(
                F=arch['F'], 
                input_size=arch['input_size'], 
                output_size=arch['output_size'], 
                stn=arch['stn']
            )
        )


class SequenceEncoderComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(SequenceEncoderComponent, self).__init__(
            from_layer, 
            to_layer, 
            RNN(
                input_pool=arch['input_pool'], 
                layers=arch['layers']
            )
        )

'''
class BrickComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(BrickComponent, self).__init__(
            from_layer, 
            to_layer, 
            build_brick(arch))
'''


class TPS_STN(nn.Module):
    def __init__(self, F, input_size, output_size, stn):
        super(TPS_STN, self).__init__()
        self.F = F
        self.input_size = input_size
        self.output_size = output_size
        self.feature_extractor = nn.Sequential(
            GVGG(stn['feature_extractor']['encoder']['backbone']['layers']),
            CollectBlock(from_layer=stn['feature_extractor']['collect']['from_layer'])
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=stn['pool']['output_size'])
        heads = []
    
        for head in stn['head']:
            heads.append(
                FCModule(
                    in_channels=head['in_channels'],
                    out_channels=head['out_channels'],
                    activation=head['activation'])
            )
        self.heads = nn.Sequential(*heads)

        self.grid_generator = GridGenerator(F, output_size)

        # Init last fc in heads
        last_fc = heads[-1].fc
        last_fc.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        last_fc.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, x):
        batch_size = x.size(0)

        batch_C_prime = self.feature_extractor(x)
        batch_C_prime = self.pool(batch_C_prime).view(batch_size, -1)
        batch_C_prime = self.heads(batch_C_prime)

        build_P_prime_reshape = self.grid_generator(batch_C_prime)

        if torch.__version__ > "1.2.0":
            out = F.grid_sample(x, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            out = F.grid_sample(x, build_P_prime_reshape, padding_mode='border')

        return out
    

class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, output_size, eps=1e-6):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = eps
        self.output_height, self.output_width = output_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.output_width, self.output_height)
        ## for multi-gpu, you need register buffer
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3
        ## for fine-tuning with different image width, you may use below instead of self.register_buffer
        #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)

        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)

        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )

        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime, device=None):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2

        return batch_P_prime  # batch_size x n x 2

    def forward(self, x):
        batch_size = x.size(0)

        build_P_prime = self.build_P_prime(x.view(batch_size, self.F, 2), x.device)  # batch_size x n (= output_width x output_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.output_height, self.output_width, 2])

        return build_P_prime_reshape
    

class GVGG(nn.Module):
    def __init__(self, layers):
        super(GVGG, self).__init__()

        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':                
                layer = ConvModule(
                    in_channels=layer_cfg['in_channels'],
                    out_channels=layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg['stride'],
                    padding=layer_cfg['padding'],
                    # dilation=post['dilation'],
                    # groups=post['groups'],
                    # bias=post['bias'],
                    # conv_cfg=post['conv_cfg'],
                    norm_cfg=layer_cfg['norm_cfg'],
                    # activation=post['activation'],
                    # inplace=post['inplace'],
                    # order=post['order'],
                    # dropout=post['dropout']
                )
            elif layer_name == 'pool':
                layer = nn.MaxPool2d(kernel_size=layer_cfg['kernel_size'], stride=layer_cfg['stride'])
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))

        init_weights(self.modules())

    def forward(self, x):
        feats = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x

        return feats
    

class FCModule(nn.Module):
    """FCModule

    Args:
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 activation='relu',
                 inplace=True,
                 dropout=None,
                 order=('fc', 'act')):
        super(FCModule, self).__init__()
        self.order = order
        self.activation = activation
        self.inplace = inplace

        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None

        self.fc = nn.Linear(in_channels, out_channels, bias)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu', 'tanh']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
            elif self.activation == 'tanh':
                self.activate = nn.Tanh()

        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.order == ('fc', 'act'):
            x = self.fc(x)

            if self.with_activatation:
                x = self.activate(x)
        elif self.order == ('act', 'fc'):
            if self.with_activatation:
                x = self.activate(x)
            x = self.fc(x)

        if self.with_dropout:
            x = self.dropout(x)

        return x


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (str or None): Config dict for activation layer.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act'),
                 dropout=None):
        super(ConvModule, self).__init__()
        assert isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              groups=groups, 
                              bias=bias)
        
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = 'bn', nn.BatchNorm2d(num_features=norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu', 'tanh', 'sigmoid']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
            elif self.activation == 'tanh':
                self.activate = nn.Tanh()
            elif self.activation == 'sigmoid':
                self.activate = nn.Sigmoid()

        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x
    

class CollectBlock(nn.Module):
    """CollectBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super(CollectBlock, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):

        if self.to_layer is None:
            if isinstance(self.from_layer, str):
                return feats[self.from_layer]
            elif isinstance(self.from_layer, list):
                return {f_layer: feats[f_layer] for f_layer in self.from_layer}
        else:
            if isinstance(self.from_layer, str):
                feats[self.to_layer] = feats[self.from_layer]
            elif isinstance(self.from_layer, list):
                feats[self.to_layer] = {f_layer: feats[f_layer] for f_layer in self.from_layer}


class RNN(nn.Module):
    def __init__(self, input_pool, layers, keep_order=False):
        super(RNN, self).__init__()
        self.keep_order = keep_order

        if input_pool:
            self.input_pool = nn.AdaptiveAvgPool2d(output_size=input_pool['output_size'])

        self.layers = nn.ModuleList()

        self.layers.add_module('{}_{}'.format(0, 'rnn'), nn.LSTM(
                input_size=layers[0][1]['input_size'], 
                hidden_size=layers[0][1]['hidden_size'], 
                batch_first=layers[0][1]['batch_first'], 
                bidirectional=layers[0][1]['bidirectional']
        ))
        self.layers.add_module('{}_{}'.format(1, 'fc'), nn.Linear(
                in_features=layers[1][1]['in_features'],
                out_features=layers[1][1]['out_features']
        ))
        self.layers.add_module('{}_{}'.format(2, 'rnn'), nn.LSTM(
                input_size=layers[2][1]['input_size'], 
                hidden_size=layers[2][1]['hidden_size'], 
                batch_first=layers[2][1]['batch_first'], 
                bidirectional=layers[2][1]['bidirectional']
        ))
        self.layers.add_module('{}_{}'.format(3, 'fc'), nn.Linear(
                in_features=layers[3][1]['in_features'],
                out_features=layers[3][1]['out_features']
        ))
        init_weights(self.modules())

    @property
    def with_input_pool(self):
        return hasattr(self, 'input_pool') and self.input_pool

    def forward(self, x):
        if self.with_input_pool:
            out = self.input_pool(x).squeeze(2)
        else:
            out = x
        # input order (B, C, T) -> (B, T, C)
        out = out.permute(0, 2, 1)
        for layer_name, layer in self.layers.named_children():

            if 'rnn' in layer_name:
                layer.flatten_parameters()
                out, _ = layer(out)
            else:
                out = layer(out)
        if not self.keep_order:
            out = out.permute(0, 2, 1).unsqueeze(2)

        return out.contiguous()
    

from torchvision.models.resnet import BasicBlock, Bottleneck

BLOCKS = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
}

class GResNet(nn.Module):
    def __init__(self, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(GResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':
                layer = ConvModule(
                    in_channels=layer_cfg['in_channels'],
                    out_channels=layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'], 
                    stride=layer_cfg['stride'], 
                    padding=layer_cfg['padding'], 
                    norm_cfg=layer_cfg['norm_cfg']
                )
                self.inplanes = layer_cfg['out_channels']
            elif layer_name == 'pool':
                layer = nn.MaxPool2d(
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg['stride'],
                    padding=layer_cfg['padding']
                )
            elif layer_name == 'block':
                layer = self._make_layer(**layer_cfg)
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))

        init_weights(self.modules())

    def _make_layer(self, block_name, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        block = BLOCKS[block_name]

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes, 
                    out_channels=planes * block.expansion,
                    kernel_size=1, 
                    stride=stride,
                    bias=False
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x

        return feats

