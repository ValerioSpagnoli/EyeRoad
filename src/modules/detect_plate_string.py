import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from ALPRNet.src.models.model_builder import ModelBuilder
from ALPRNet.src.utils.data_utils import get_vocabularies
from ALPRNet.src.utils.pred_utils import predict_code


class StringPlateDetector:
    def __init__(self, weights_path=None):
        self.char2id, self.id2char = get_vocabularies()
        self.max_len = 15
        self.model = ModelBuilder(rec_num_classes=len(self.char2id), sDim=512, attDim=512, max_len_labels=self.max_len, eos=self.char2id['EOS'], STN_ON=True)
        weights = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(weights)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor()
        ])
        
        
    def detect_plate_string(self, image=None):
        img = np.stack([image]*3, axis=-1)
        pil_image = Image.fromarray(img)
        plate_string = predict_code(self.model, pil_image, self.max_len, self.char2id['EOS'], self.transform, 'cpu')
        return plate_string