import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.weight_init import init_weights

from .components import RectificatorComponent, FeatureExtractorComponent, SequenceEncoderComponent, CollectBlock

class GModel(nn.Module):
    def __init__(self, body, head, need_text=True):
        super(GModel, self).__init__()
        self.body = GBody(
            pipelines = body['pipelines'], 
            collect = body['collect']
        )
        self.head = AttHead(
            cell = head['cell'],
            generator = head['generator'],
            num_steps = head['num_steps'],
            num_class = head['num_class'],
            input_attention_block = head['input_attention_block'],
            output_attention_block = head['output_attention_block'],
            text_transform = head['text_transform'],
            holistic_input_from = head['holistic_input_from']
        )
        self.need_text = need_text

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        x = self.body(inputs[0])

        if self.need_text:
            out = self.head(x, inputs[1])
        else:
            out = self.head(x)

        return out
    


#### BODY

class GBody(nn.Module):
    def __init__(self, pipelines, collect=None):
        super(GBody, self).__init__()

        self.input_to_layer = 'input'
        self.components = nn.ModuleList([
            RectificatorComponent(
                pipelines[0]['from_layer'],
                pipelines[0]['to_layer'],
                pipelines[0]['arch']),
            FeatureExtractorComponent(
                pipelines[1]['from_layer'],
                pipelines[1]['to_layer'],
                pipelines[1]['arch']), 
            SequenceEncoderComponent(
                pipelines[2]['from_layer'],
                pipelines[2]['to_layer'],
                pipelines[2]['arch'])
        ])

        if collect is not None:
            self.collect = CollectBlock(from_layer=collect['from_layer'], 
                                        to_layer=collect['to_layer'])

    @property
    def with_collect(self):
        return hasattr(self, 'collect') and self.collect is not None

    def forward(self, x):
        feats = {self.input_to_layer: x}

        for component in self.components:
            component_from = component.from_layer
            component_to = component.to_layer

            if isinstance(component_from, list):
                inp = {key: feats[key] for key in component_from}
                out = component(**inp)
            else:
                inp = feats[component_from]
                out = component(inp)
            feats[component_to] = out

        if self.with_collect:
            return self.collect(feats)
        else:
            return feats

### HEAD

class AttHead(nn.Module):
    def __init__(self,
                 cell,
                 generator,
                 num_steps,
                 num_class,
                 input_attention_block=None,
                 output_attention_block=None,
                 text_transform=None,
                 holistic_input_from=None):
        super(AttHead, self).__init__()

        if input_attention_block is not None:
            self.input_attention_block = CellAttentionBlock(feat=input_attention_block['feat'], 
                                                            hidden=input_attention_block['hidden'], 
                                                            fusion_method=input_attention_block['fusion_method'], 
                                                            post=input_attention_block['post'], 
                                                            post_activation=input_attention_block['post_activation'])

        self.cell = LSTMCell(input_size=cell['input_size'], hidden_size=cell['hidden_size'])
        self.generator = nn.Linear(in_features=generator['in_features'], out_features=generator['out_features']) #, bias=generator['bias'])
        self.num_steps = num_steps
        self.num_class = num_class

        if output_attention_block is not None:
            self.output_attention_block = None # build_brick(output_attention_block)

        if text_transform is not None:
            self.text_transform = None # build_torch_nn(text_transform)

        if holistic_input_from is not None:
            self.holistic_input_from = holistic_input_from

        self.register_buffer('embeddings', torch.diag(torch.ones(self.num_class)))
        init_weights(self.modules())

    @property
    def with_holistic_input(self):
        return hasattr(self, 'holistic_input_from') and self.holistic_input_from

    @property
    def with_input_attention(self):
        return hasattr(self, 'input_attention_block') and self.input_attention_block is not None

    @property
    def with_output_attention(self):
        return hasattr(self, 'output_attention_block') and self.output_attention_block is not None

    @property
    def with_text_transform(self):
        return hasattr(self, 'text_transform') and self.text_transform

    def forward(self, feats, texts):
        batch_size = texts.size(0)

        hidden = self.cell.init_hidden(batch_size, device=texts.device)
        if self.with_holistic_input:
            holistic_input = feats[self.holistic_input_from][:, :, 0, -1]
            hidden = self.cell(holistic_input, hidden)

        out = []

        if self.training:
            use_gt = True
            assert self.num_steps == texts.size(1)
        else:
            use_gt = False
            # assert texts.size(1) == 1 ################################

        for i in range(self.num_steps):
            if i == 0:
                indexes = texts[:, i]
            else:
                if use_gt:
                    indexes = texts[:, i]
                else:
                    _, indexes = out[-1].max(1)
            text_feat = self.embeddings.index_select(0, indexes)

            if self.with_text_transform:
                text_feat = self.text_transform(text_feat)

            if self.with_input_attention:
                attention_feat = self.input_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                cell_input = torch.cat([attention_feat, text_feat], dim=1)
            else:
                cell_input = text_feat
            hidden = self.cell(cell_input, hidden)
            out_feat = self.cell.get_output(hidden)

            if self.with_output_attention:
                attention_feat = self.output_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                out_feat = torch.cat([self.cell.get_output(hidden), attention_feat], dim=1)

            out.append(self.generator(out_feat))

        out = torch.stack(out, dim=1)

        return out


class CellAttentionBlock(nn.Module):
    def __init__(self, feat, hidden, fusion_method='add', post=None, post_activation='softmax'):
        super(CellAttentionBlock, self).__init__()

        feat_ = feat.copy()
        self.feat_from = feat_.pop('from_layer')
        self.feat_block = ConvModule(
                in_channels=feat['in_channels'],
                out_channels=feat['out_channels'],
                kernel_size=feat['kernel_size'],
                # stride=feat['stride'],
                # padding=feat['padding'],
                # dilation=feat['dilation'],
                # groups=feat['groups'],
                bias=feat['bias'],
                # conv_cfg=feat['conv_cfg'],
                # norm_cfg=feat['norm_cfg'],
                activation=feat['activation'],
                # inplace=feat['inplace'],
                # order=feat['order'],
                # dropout=feat['dropout']
            )
        
        self.hidden_block = ConvModule(
                in_channels=hidden['in_channels'],
                out_channels=hidden['out_channels'],
                kernel_size=hidden['kernel_size'],
                # stride=hidden['stride'],
                # padding=hidden['padding'],
                # dilation=hidden['dilation'],
                # groups=hidden['groups'],
                # bias=hidden['bias'],
                # conv_cfg=hidden['conv_cfg'],
                # norm_cfg=hidden['norm_cfg'],
                activation=hidden['activation'],
                # inplace=hidden['inplace'],
                # order=hidden['order'],
                # dropout=hidden['dropout']
            )

        self.fusion_method = fusion_method
        self.activate = post_activation

        if post is not None:
            self.post_block = ConvModule(
                in_channels=post['in_channels'],
                out_channels=post['out_channels'],
                kernel_size=post['kernel_size'],
                # stride=post['stride'],
                # padding=post['padding'],
                # dilation=post['dilation'],
                # groups=post['groups'],
                bias=post['bias'],
                # conv_cfg=post['conv_cfg'],
                # norm_cfg=post['norm_cfg'],
                activation=post['activation'],
                # inplace=post['inplace'],
                # order=post['order'],
                # dropout=post['dropout']
            )
        else:
            self.post_block = nn.Sequential()

    def forward(self, feats, hidden):
        feat = feats[self.feat_from]
        b, c = feat.size(0), feat.size(1)
        feat_to_attend = feat.view(b, c, -1)

        x = self.feat_block(feat)
        y = self.hidden_block(hidden)

        assert self.fusion_method in ['add', 'dot']
        if self.fusion_method == 'add':
            attention_map = x + y
        elif self.fusion_method == 'dot':
            attention_map = x * y

        attention_map = self.post_block(attention_map)
        b, c = attention_map.size(0), attention_map.size(1)
        attention_map = attention_map.view(b, c, -1)

        assert self.activate in ['softmax', 'sigmoid']
        if self.activate == 'softmax':
            attention_map = F.softmax(attention_map, dim=2)
        elif self.activate == 'sigmoid':
            attention_map = F.sigmoid(attention_map)

        feat = feat_to_attend * attention_map
        feat = feat.sum(2)

        return feat
    

########

class BaseCell(nn.Module):
    def __init__(self, basic_cell, input_size, hidden_size, bias=True, num_layers=1):
        super(BaseCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(basic_cell(input_size=input_size, hidden_size=hidden_size, bias=bias))
            else:
                self.cells.append(basic_cell(input_size=hidden_size, hidden_size=hidden_size, bias=bias))
        init_weights(self.modules())

    def init_hidden(self, batch_size, device=None, value=0):
        raise NotImplementedError()

    def get_output(self, hiddens):
        raise NotImplementedError()

    def get_hidden_state(self, hidden):
        raise NotImplementedError()

    def forward(self, x, pre_hiddens):
        next_hiddens = []

        hidden = None
        for i, cell in enumerate(self.cells):
            if i == 0:
                hidden = cell(x, pre_hiddens[i])
            else:
                hidden = cell(self.get_hidden_state(hidden), pre_hiddens[i])

            next_hiddens.append(hidden)

        return next_hiddens


class LSTMCell(BaseCell):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1):
        super(LSTMCell, self).__init__(nn.LSTMCell, input_size, hidden_size, bias, num_layers)

    def init_hidden(self, batch_size, device=None, value=0):
        hiddens = []
        for _ in range(self.num_layers):
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_size).fill_(value).to(device),
                torch.FloatTensor(batch_size, self.hidden_size).fill_(value).to(device),
            )
            hiddens.append(hidden)

        return hiddens

    def get_output(self, hiddens):
        return hiddens[-1][0]

    def get_hidden_state(self, hidden):
        return hidden[0]


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