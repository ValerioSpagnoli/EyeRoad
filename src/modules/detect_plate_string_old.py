# import pytesseract as pt
import torch
import torchvision.transforms as transforms
import os
import string
from PIL import Image

from ANPR.src.models.model import GModel
from ANPR.src.utils.postprocess import postprocess
from ANPR.src.utils.attn_converter import AttnConverter


class StringPlateDetector:
    def __init__(self):
        self.model = load_anpr(device='cpu', weights_dir='../weights/anpr_weights/')
        
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),      # Bright, contr, ...
            transforms.Grayscale(num_output_channels=1),                                        # ToGray
            transforms.RandomRotation(degrees=10),                                              # Rand. rot.
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),                          # Rand. pers.
            transforms.Resize((32, 100)),                                                       # Resize
            transforms.ToTensor(),                                                              # ToTensor
            transforms.Normalize(mean=0.5, std=0.5),                                            # Normalize
        ])
        
        self.converter = AttnConverter(character=string.ascii_uppercase+string.digits, batch_max_length=25)
        
    def detect_plate_string(self, plate_img=None, weights_dir=None):
        self.model = load_anpr(device='cpu', weights_dir=weights_dir)
        self.model.eval()
    
        image = self.transform(Image.fromarray(plate_img)).unsqueeze(0)
        pred = self.model((image.to('cpu'), torch.tensor([[0]]).to('cpu')))
        pred, prob = postprocess(pred, self.converter, None)
        return (pred[0], prob[0].item())



def load_anpr(device='cpu', weights_dir=None):

    # Define args
    args = {
        'need_text': True, 
        'body': {
            'type': 'GBody', 
            'pipelines': [
                {'type': 'RectificatorComponent', 'from_layer': 'input', 'to_layer': 'rect', 
                    'arch': {'type': 'TPS_STN', 'F': 20, 'input_size': (32, 100), 'output_size': (32, 100), 'stn': {'feature_extractor': {'encoder': { 'backbone': {'type': 'GVGG', 'layers': [('conv', {'type': 'ConvModule', 'in_channels': 1, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2}), ('conv', {'type': 'ConvModule', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2}), ('conv', {'type': 'ConvModule', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2}), ('conv', {'type': 'ConvModule', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}})]}}, 'collect': {'type': 'CollectBlock', 'from_layer': 'c3'}}, 'pool': {'type': 'AdaptiveAvgPool2d', 'output_size': 1}, 'head': [{'type': 'FCModule', 'in_channels': 512, 'out_channels': 256, 'activation': 'relu'}, {'type': 'FCModule', 'in_channels': 256, 'out_channels': 40, 'activation': None}]}}}, 
                {'type': 'FeatureExtractorComponent', 'from_layer': 'rect', 'to_layer': 'cnn_feat', 
                    'arch': {'encoder': {'backbone': {'type': 'GResNet', 'layers': [('conv', {'type': 'ConvModule', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('conv', {'type': 'ConvModule', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 0}), ('block', {'block_name': 'BasicBlock', 'planes': 128, 'blocks': 1, 'stride': 1}), ('conv', {'type': 'ConvModule', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 0}), ('block', {'block_name': 'BasicBlock', 'planes': 256, 'blocks': 2, 'stride': 1}), ('conv', {'type': 'ConvModule', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('pool', {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': (2, 1), 'padding': (0, 1)}), ('block', {'block_name': 'BasicBlock', 'planes': 512, 'blocks': 5, 'stride': 1}), ('conv', {'type': 'ConvModule', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'norm_cfg': {'type': 'BN'}}), ('block', {'block_name': 'BasicBlock', 'planes': 512, 'blocks': 3, 'stride': 1}), ('conv', {'type': 'ConvModule', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': (2, 1), 'padding': (0, 1), 'norm_cfg': {'type': 'BN'}}), ('conv', {'type': 'ConvModule', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 1, 'padding': 0, 'norm_cfg': {'type': 'BN'}})]}}, 'collect': {'type': 'CollectBlock', 'from_layer': 'c4'}}}, 
                {'type': 'SequenceEncoderComponent', 'from_layer': 'cnn_feat', 'to_layer': 'rnn_feat', 
                    'arch': {'type': 'RNN', 'input_pool': {'type': 'AdaptiveAvgPool2d', 'output_size': (1, None)}, 'layers': [('rnn', {'type': 'LSTM', 'input_size': 512, 'hidden_size': 256, 'bidirectional': True, 'batch_first': True}), ('fc', {'type': 'Linear', 'in_features': 512, 'out_features': 256}), ('rnn', {'type': 'LSTM', 'input_size': 256, 'hidden_size': 256, 'bidirectional': True, 'batch_first': True}), ('fc', {'type': 'Linear', 'in_features': 512, 'out_features': 256})]}}
            ],
            'collect': None
        }, 
        'head': {
            'type': 'AttHead', 
            'num_class': 38, 
            'num_steps': 26, 
            'cell': {'type': 'LSTMCell', 'input_size': 294, 'hidden_size': 256}, 
            'input_attention_block': {'type': 'CellAttentionBlock', 'feat': {'from_layer': 'rnn_feat', 'type': 'ConvModule', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 1, 'bias': False, 'activation': None}, 'hidden': {'type': 'ConvModule', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 1, 'activation': None}, 'fusion_method': 'add', 'post': {'type': 'ConvModule', 'in_channels': 256, 'out_channels': 1, 'kernel_size': 1, 'bias': False, 'activation': 'tanh', 'order': ('act', 'conv', 'norm')}, 'post_activation': 'softmax'}, 
            'output_attention_block': None,
            'text_transform': None,
            'holistic_input_from': None,
            'generator': {'type': 'Linear', 'in_features': 256, 'out_features': 38}
        }
    }

    # Define model
    model = GModel(body=args['body'], head=args['head'], need_text=args['need_text'])
    # Move model to gpt
    model.to(device)
    # Load weights
    weights = torch.load(os.path.join(weights_dir, f'newdata_100ep_v2_train1.pth.zip'), map_location=device)
    model.load_state_dict(weights)
    return model