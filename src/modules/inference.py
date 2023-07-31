import torch
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
    boxes_t = boxes[want_st].cpu().detach().numpy()
    labels_t = labels[want_st].cpu().detach().numpy()
    scores_t = scores[want_st].cpu().detach().numpy()
                
    if(len(boxes_t)==0):
        boxes = [boxes.cpu().detach().numpy()[0]]
        labels = [labels.cpu().detach().numpy()[0]]
        scores = [scores.cpu().detach().numpy()[0]]
    else:
        boxes = boxes_t
        labels = labels_t
        scores = scores_t
        
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
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
        
    if(len(boxes)==0):
        return (None, None, None)
    
    want_st = scores > score_threshold
    boxes_t = boxes[want_st].cpu().detach().numpy()
    labels_t = labels[want_st].cpu().detach().numpy()
    scores_t = scores[want_st].cpu().detach().numpy()
            
    if(len(boxes_t)==0):
        boxes = [boxes.cpu().detach().numpy()[0]]
        labels = [labels.cpu().detach().numpy()[0]]
        scores = [scores.cpu().detach().numpy()[0]]
    else:
        boxes = boxes_t
        labels = labels_t
        scores = scores_t
        
    i=0
    while i < len(labels):
        label = labels[i]
        if label != 2: # != plates
            boxes = np.delete(boxes, i, axis=0)
            labels = np.delete(labels, i, axis=0)
            scores = np.delete(scores, i, axis=0)
        else: i+=1
        
    return (boxes, labels, scores)