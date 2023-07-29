import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.inference import decode_prediction_vehicles, decode_prediction_plates
from modules.detect_plate_string import detect_plate_string


# Plot one image prediction
def plot_one_image(model=None, img_param=None, sr_weights_path=3, cv2window=False, cv2imshow=False, plate_plot=False, cv2_vehicles_cfg=None, cv2_plates_cfg=None):
    
    if cv2_vehicles_cfg is None:
        cv2_vehicles_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'color':(0,0,255), 'thickness':2, 'lineType':cv2.LINE_AA}
    if cv2_plates_cfg is None:
        cv2_plates_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'color':(0,255,0), 'thickness':2, 'lineType':cv2.LINE_AA}
    
    if cv2window: cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)   
    
    if(isinstance(img_param, str)):
        img = cv2.imread(img_param)
    elif(isinstance(img_param, np.ndarray)):
        img = img_param
        
    if img is None:
        print('Error: unable to read image.')
        return
                
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img)
    tensor_img = torch.reshape(tensor_img, (1, tensor_img.shape[0], tensor_img.shape[1], tensor_img.shape[2]))    
    
    predictions_vehicles = model(tensor_img)[0]
        
    boxes_vehicles, _, scores_vehicles = decode_prediction_vehicles(prediction=predictions_vehicles)
        
    new_img = img
    
    if boxes_vehicles is not None:    
        for i in range(len(boxes_vehicles)):
            box_vehicle = boxes_vehicles[i]
            score_vehicle = scores_vehicles[i]

            xmin_vehicle, xmax_vehicle = (box_vehicle[0]).astype(int), (box_vehicle[2]).astype(int)
            ymin_vehicle, ymax_vehicle = (box_vehicle[1]).astype(int), (box_vehicle[3]).astype(int)
                
            new_img = cv2.rectangle(new_img, (xmin_vehicle, ymin_vehicle), (xmax_vehicle, ymax_vehicle), cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'])
            new_img = cv2.putText(new_img, f"vehicle {score_vehicle:.2f}", (xmin_vehicle, ymin_vehicle - 5), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])
                
            # crop the image on detected vehicle
            vehicle_image = tensor_img.squeeze(dim=0)[:, ymin_vehicle:ymax_vehicle, xmin_vehicle:xmax_vehicle].unsqueeze(dim=0)  
                
            # recompute the model on the cropped image
            predictions_plates = model(vehicle_image)[0]
                
            # decode predictions of plates
            boxes_plates, _, scores_plates = decode_prediction_plates(prediction=predictions_plates)
                
            if boxes_plates is not None:
                for i in range(len(boxes_plates)):
                    box_plate = boxes_plates[i]
                    score_plate = scores_plates[i]
                                
                    xmin_plate, xmax_plate = (box_plate[0]).astype(int), (box_plate[2]).astype(int)
                    ymin_plate, ymax_plate = (box_plate[1]).astype(int), (box_plate[3]).astype(int)
                    
                    roi = img[ymin_vehicle+ymin_plate:ymin_vehicle+ymax_plate, xmin_vehicle+xmin_plate:xmin_vehicle+xmax_plate]
                            
                    if sr_weights_path is not None:
                        sr = cv2.dnn_superres.DnnSuperResImpl_create()
                        sr.readModel(sr_weights_path)
                        sr.setModel("edsr",int(sr_weights_path.split('x')[1].split('.')[0]))
                        roi = sr.upsample(roi)  
                        
                        
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
                    plate_string = detect_plate_string(plate_img=binary_roi)
                    
                    
                    if plate_plot:                   
                        _, ax = plt.subplots(1, 3, figsize=(10,5))   
                        
                        ax[0].imshow(roi)
                        ax[0].set_title('roi')
                        ax[0].grid(False)
                        ax[1].imshow(gray_roi, cmap='gray')
                        ax[1].set_title('gray_roi')
                        ax[1].grid(False)
                        ax[2].imshow(binary_roi, cmap='gray')
                        ax[2].set_title('binary_roi')    
                        ax[2].grid(False)
                                                            
                        print(f'plate_string: {plate_string}')                    
                    
                    new_img = cv2.rectangle(new_img, (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate), (xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), cv2_plates_cfg['color'], cv2_plates_cfg['thickness'])
                    new_img = cv2.putText(new_img, f"plate {score_plate:.2f}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate - 5), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType']) 
                    new_img = cv2.putText(new_img, f"{plate_string}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymax_plate + 20), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType'])    
                        
            if cv2imshow: cv2.imshow('Video', new_img)
    
    else:
        if cv2imshow: cv2.imshow('Video', img)
    

    if cv2window: cv2.waitKey(0)
    if cv2window: cv2.destroyAllWindows()
    
    

def real_time_object_detector(model=None, video_path=None, sr_weights_path=None, cv2_vehicles_cfg=None, cv2_plates_cfg=None):
        
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    
    while True:
        ret, frame = cap.read()
        
        if ret == False:
            print('Unable to read video')
            break
            
        plot_one_image(model=model, img_param=frame, sr_weights_path=sr_weights_path, cv2window=False, cv2imshow=True, plate_plot=False, cv2_vehicles_cfg=cv2_vehicles_cfg, cv2_plates_cfg=cv2_plates_cfg)    
                    
        if cv2.waitKey(30) == 27 :
            break
            
    cv2.destroyAllWindows()
    cap.release()