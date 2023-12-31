import torch
import torchvision
import numpy as np
 
from modules.model import get_plateDetectorModel


def load_model_for_inference(weights_file=None, device=None):   
    model = get_plateDetectorModel()
    model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))
    model.eval()
    
    return model


# Takes decodes this dictionary, and filters out any predictions with a score lower than score_threshold.     
def decode_prediction_vehicles(prediction=None, score_threshold=0.3):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    
    if(len(boxes)==0):
        return (None, None, None)
    
    want_st = scores > score_threshold
    boxes_t = boxes[want_st]
    labels_t = labels[want_st]
    scores_t = scores[want_st]
    
    if(len(boxes_t)==0):
        boxes = [boxes.cpu().detach().numpy()[0]]
        labels = [labels.cpu().detach().numpy()[0]]
        scores = [scores.cpu().detach().numpy()[0]]
    else:           
        want_nms = torchvision.ops.nms(boxes=boxes_t, scores=scores_t, iou_threshold=0.2)
        
        boxes_nms = boxes[want_nms].cpu().detach().numpy()
        labels_nms = labels[want_nms].cpu().detach().numpy()
        scores_nms = scores[want_nms].cpu().detach().numpy()
        
        if(len(boxes_nms)==0):
            boxes = [boxes.cpu().detach().numpy()[0]]
            labels = [labels.cpu().detach().numpy()[0]]
            scores = [scores.cpu().detach().numpy()[0]]
        else:
            boxes = boxes_nms
            labels = labels_nms
            scores = scores_nms
          
    i=0
    while i < len(labels):
        label = labels[i]
        if label != 1: # != vechiles
            boxes = np.delete(boxes, i, axis=0)
            labels = np.delete(labels, i, axis=0)
            scores = np.delete(scores, i, axis=0)
        else: i+=1
        
    return (boxes, labels, scores)


# we know that in a cropped image of a vehicle must be only one plate, so we take the prediction with higher score
def decode_prediction_plates(prediction=None, score_threshold=0.7):
    boxes = prediction["boxes"].cpu().detach().numpy()
    scores = prediction["scores"].cpu().detach().numpy()
    labels = prediction["labels"].cpu().detach().numpy()
                
    i=0
    while i < len(labels):
        label = labels[i]
        if label != 2: # != plates
            boxes = np.delete(boxes, i, axis=0)
            labels = np.delete(labels, i, axis=0)
            scores = np.delete(scores, i, axis=0)
        else: i+=1
    
    if(len(boxes)==0 or scores[0]<score_threshold):
        return (None, None, None)
    
    return (boxes[0], labels[0], scores[0])