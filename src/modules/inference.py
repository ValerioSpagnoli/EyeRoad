import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
 
from modules.model import get_plateDetectorModel


def load_model_for_inference(weights_dir=None, device=None):   
    model = get_plateDetectorModel()
    model.load_state_dict(torch.load(weights_dir+"detector.pt", map_location=torch.device(device)))
    model.eval()
    
    return model


# Unbatch a dataloader batch
def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y



# Return the predictions for one batch
def predict_batch(batch=None, model=None, device='cpu'):
    
    model.to(device)
    model.eval()

    X, _ = unbatch(batch, device = device)
    predictions = model(X)

    return [x.cpu() for x in X], predictions

# Peform prediction of data_loader
def predict(model=None, data_loader=None, device='cpu'):

    images = []
    predictions = [] 

    for i, batch in enumerate(data_loader):
        imgs, preds = predict_batch(batch, model, device)

        for i in range(len(imgs)):
            images.append(imgs[i])
            predictions.append(preds[i])
    
    return images, predictions


# Takes decodes this dictionary, and filters out any predictions with a score lower than score_threshold.     
def decode_prediction_vehicles(prediction=None, score_threshold = 0.3, nms_iou_threshold = 0.2):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    
    if(len(boxes)==0):
        return (None, None, None)
    
    want_st = scores > score_threshold
    boxes_t = boxes[want_st].cpu().detach().numpy()
    labels_t = labels[want_st].cpu().detach().numpy()
    scores_t = scores[want_st].cpu().detach().numpy()
    
    # want_nms = torchvision.ops.nms(boxes = boxes_t, scores = scores_t, iou_threshold = nms_iou_threshold)
    # boxes_nms = boxes_t[want_nms].cpu().detach().numpy()
    # labels_nms = labels_t[want_nms].cpu().detach().numpy()
    # scores_nms = scores_t[want_nms].cpu().detach().numpy()
            
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
def decode_prediction_plates(prediction=None, score_threshold = 0.7):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
        
    if(len(boxes)==0):
        return (None, None, None)
    
    want_st = scores > score_threshold
    boxes_t = boxes[want_st].cpu().detach().numpy()
    labels_t = labels[want_st].cpu().detach().numpy()
    scores_t = scores[want_st].cpu().detach().numpy()
    
    # want_nms = torchvision.ops.nms(boxes = boxes_t, scores = scores_t, iou_threshold = nms_iou_threshold)
    # boxes_nms = boxes_t[want_nms].cpu().detach().numpy()
    # labels_nms = labels_t[want_nms].cpu().detach().numpy()
    # scores_nms = scores_t[want_nms].cpu().detach().numpy()
            
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


# Plot a grid of images predictions
def plot_grid_images(images=None, predictions=None, grid_shape=(1,2), figsize=(10, 5)):

    num_images = len(images)
    grid_shape = (num_images, 1)
    rows, cols = grid_shape

    if num_images > rows * cols:
        print(f"Warning: Number of images exceeds grid size. Only the first {rows * cols} images will be shown.")
    
    fig, axes = plt.subplots(rows, cols, figsize=(3,3*num_images))
    
    for i, ax in enumerate(axes.ravel()):
        img = images[i]
        pred = predictions[i]
        
        img = img.numpy().transpose((1, 2, 0))
        boxes, labels, scores = decode_prediction_vehicles(pred)
        
        ax.imshow(img)
        ax.axis('off')

        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
            
                xmin = (box[0]).astype(np.int32)
                ymin = (box[1]).astype(np.int32)
                xmax = (box[2]).astype(np.int32)
                ymax = (box[3]).astype(np.int32)
                            
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
                ax.text(xmin, ymin - 5, f'Score: {score:.2f}', fontsize=10, color='red')
                ax.add_patch(rect)
                

    plt.tight_layout()
    plt.show()