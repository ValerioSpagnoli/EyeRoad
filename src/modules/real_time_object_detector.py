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
    
    _, ax = plt.subplots(1, 4, figsize=(10,5))  
    
        
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
        
    new_img = img.copy()
    
    xmin_detection_area_plates = 30
    ymin_detection_area_plates = int(img.shape[0]/10)*6
    xmax_detection_area_plates = int(img.shape[1])-30
    ymax_detection_area_plates = int(img.shape[0])-30
    
    xmin_detection_area_vehicles = 30
    ymin_detection_area_vehicles = int(img.shape[0]/10)*4
    xmax_detection_area_vehicles = int(img.shape[1])-30
    ymax_detection_area_vehicles = int(img.shape[0])-30
        
    cv2.rectangle(new_img, (xmin_detection_area_vehicles, ymin_detection_area_vehicles), (xmax_detection_area_vehicles, ymax_detection_area_vehicles), (0,150,150), 2)
    cv2.rectangle(new_img, (xmin_detection_area_plates, ymin_detection_area_plates), (xmax_detection_area_plates, ymax_detection_area_plates), (0,150,150), 2)
    
    if cv2imshow: cv2.imshow('Video', new_img)
    
    if boxes_vehicles is not None:    
        for i in range(len(boxes_vehicles)):
            box_vehicle = boxes_vehicles[i]
            score_vehicle = scores_vehicles[i]

            xmin_vehicle, xmax_vehicle = (box_vehicle[0]).astype(int), (box_vehicle[2]).astype(int)
            ymin_vehicle, ymax_vehicle = (box_vehicle[1]).astype(int), (box_vehicle[3]).astype(int)
            
            if(ymin_vehicle < ymin_detection_area_vehicles):
                continue
                
            new_img = cv2.rectangle(new_img, (xmin_vehicle, ymin_vehicle), (xmax_vehicle, ymax_vehicle), cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'])
            new_img = cv2.putText(new_img, f"vehicle {score_vehicle:.2f}", (xmin_vehicle, ymin_vehicle - 5), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])
            if cv2imshow: cv2.imshow('Video', new_img)
            
                        
            box_plate_area = new_img.copy()   
            xmin_box_plate_area = xmin_vehicle + int((xmax_vehicle - xmin_vehicle)/4)
            xmax_box_plate_area = xmin_vehicle + int(((xmax_vehicle - xmin_vehicle)/4) * 3)
            ymin_box_plate_area = ymin_vehicle + int(((ymax_vehicle - ymin_vehicle)/10) * 4) 
            ymax_box_plate_area = ymin_vehicle + int(((ymax_vehicle - ymin_vehicle)/10) * 9)
            
            if(ymin_box_plate_area < ymin_detection_area_plates or ymax_box_plate_area > ymax_detection_area_plates or 
               xmin_box_plate_area < xmin_detection_area_plates or xmax_box_plate_area > xmax_detection_area_plates):
                continue

            cv2.rectangle(box_plate_area, (xmin_box_plate_area, ymin_box_plate_area), (xmax_box_plate_area, ymax_box_plate_area), (225,255,0), -1)
            cv2.addWeighted(box_plate_area, 0.3, new_img, 1 - 0.3, 0, new_img)
            
            #cv2.imwrite(f'{i}.jpeg', img[ymin_vehicle:ymax_vehicle, xmin_vehicle:xmax_vehicle])
                                        
            ax[0].imshow(cv2.cvtColor(img[ymin_vehicle:ymax_vehicle, xmin_vehicle:xmax_vehicle], cv2.COLOR_BGR2RGB))
            ax[0].set_title('img')     
                                            
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
                    
                    overlap_percentage = compute_overlap_percentage((xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate, xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), 
                                                                    (xmin_box_plate_area, ymin_box_plate_area, xmax_box_plate_area, ymax_box_plate_area))
                    
                    area_percentage = compute_plate_area_percentage((xmin_plate, ymin_plate, xmax_plate, ymax_plate), 
                                                                    (xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle))
                    
                        
                    print('----------------------------------------------------------------------')
                    
                    if(overlap_percentage < 80 or (area_percentage < 100 or area_percentage > 400)):
                        print(f'Overlap percentage: {overlap_percentage}')
                        print(f'Area percentage (px): {area_percentage}')
                        continue
                    
                    roi = img[ymin_vehicle+ymin_plate:ymin_vehicle+ymax_plate, xmin_vehicle+xmin_plate:xmin_vehicle+xmax_plate]
                            
                    if sr_weights_path is not None:
                        sr = cv2.dnn_superres.DnnSuperResImpl_create()
                        sr.readModel(sr_weights_path)
                        sr.setModel("edsr",int(sr_weights_path.split('x')[1].split('.')[0]))
                        roi = sr.upsample(roi)
                        
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    total_pixels = gray_roi.shape[0] * gray_roi.shape[1]
                    below_threshold_pixels = np.sum(gray_roi < 160)
                    percentage_below_threshold = (below_threshold_pixels / total_pixels) * 100
                    
                    below_threshold_mask = gray_roi < 160
                    avg_distance2threshold = np.sum(160-gray_roi[below_threshold_mask])/total_pixels
                                        
                    if(percentage_below_threshold > 50):
                        gray_roi = gray_roi + int(avg_distance2threshold)+25
                        gray_roi = np.clip(gray_roi, 0, 255).astype(np.uint8)

                    _, binary_roi = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
                    plate_string = detect_plate_string(plate_img=binary_roi)
                    
                    
                    if plate_plot:                   
                        ax[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        ax[1].set_title('roi')
                        ax[2].imshow(gray_roi, cmap='gray')
                        ax[2].set_title('gray_roi')
                        ax[3].imshow(binary_roi, cmap='gray')
                        ax[3].set_title('binary_roi')     
                                                            
                    print(f'plate_string: {plate_string}')                    
                    
                    cv2.rectangle(new_img, (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate), (xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), cv2_plates_cfg['color'], cv2_plates_cfg['thickness'])
                    cv2.putText(new_img, f"plate {score_plate:.2f}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate - 5), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType']) 
                    cv2.putText(new_img, f"{plate_string}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymax_plate + 30), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType'])    
                        
            if cv2imshow: cv2.imshow('Video', new_img)
    
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
    




def compute_overlap_percentage(rect1, rect2):
    # Calculate the coordinates of the intersection rectangle
    xmin_intersection = max(rect1[0], rect2[0])
    ymin_intersection = max(rect1[1], rect2[1])
    xmax_intersection = min(rect1[2], rect2[2])
    ymax_intersection = min(rect1[3], rect2[3])

    # Calculate the area of the intersection rectangle
    width_intersection = max(0, xmax_intersection - xmin_intersection)
    height_intersection = max(0, ymax_intersection - ymin_intersection)
    area_intersection = width_intersection * height_intersection

    # Calculate the area of each rectangle
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    overlap_percentage = (area_intersection / min(area_rect1, area_rect2)) * 100

    return overlap_percentage


def compute_plate_area_percentage(plate, vehicle):
    xmin_plate, ymin_plate, xmax_plate, ymax_plate = plate[0], plate[1], plate[2], plate[3]
    xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle = vehicle[0], vehicle[1], vehicle[2], vehicle[3]
    
    width_plate = xmax_plate-xmin_plate
    height_plate = ymax_plate-ymin_plate
    width_vechile = xmax_vehicle-xmin_vehicle
    height_vehicle = ymax_vehicle-ymin_vehicle
    width_percentage = (width_plate/width_vechile) * 100
    height_percentage = (height_plate/height_vehicle) * 100
    area_percentage = width_percentage * height_percentage
    
    return area_percentage