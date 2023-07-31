from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Pretrained model for object detection
def get_plateDetectorModel(num_classes=3, feature_extraction=True):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    if feature_extraction:
        for p in model.parameters():
            p.requires_grad = False

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,num_classes)

    return model