import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.sort import Sort

from modules.inference import decode_prediction_vehicles, decode_prediction_plates
from modules.detect_plate_string import StringPlateDetector
from utils.frames2video import frames2video


# IMAGE PROCESSING
def plot_one_image(model=None, img_param=None, sr_weights_path=3, string_plate_detector=None, tracker=None, cv2window=False, cv2imshow=False, plt_plot=False, cv2_vehicles_cfg=None, cv2_plates_cfg=None):
            
    detections=[]        
            
    # if configurations of plot are not passed as paramenters, set them
    if cv2_vehicles_cfg is None:
        cv2_vehicles_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(0,0,255), 'thickness':2, 'lineType':cv2.LINE_AA}
    if cv2_plates_cfg is None:
        cv2_plates_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(255,0,0), 'thickness':2, 'lineType':cv2.LINE_AA}
    
    if cv2window: cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)   
    if plt_plot: _, ax = plt.subplots(1, 4, figsize=(10,5))  
    
    # check if the image passed as parameter is an cv2 image, or a string (i.e. the path of the image)    
    if(isinstance(img_param, str)):
        img = cv2.imread(img_param)
    elif(isinstance(img_param, np.ndarray)):
        img = img_param
        
    if img is None:
        print('Error: unable to read image.')
        return
    
    # trasform the image to a tensor and reshape it in order to have [1, c, h, w]           
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img)
    tensor_img = torch.reshape(tensor_img, (1, tensor_img.shape[0], tensor_img.shape[1], tensor_img.shape[2]))    
    
    # make inference on the model using the current image
    predictions_vehicles = model(tensor_img)[0]
        
    # decode the predicitons done on the image
    boxes_vehicles, _, scores_vehicles = decode_prediction_vehicles(prediction=predictions_vehicles)
    
    # the variable new_img will be the image on which will be done the cv2 plots
    new_img = img.copy()
    
    # define the area inside which will be consider the detections of vehicles
    xmin_detection_area_vehicles = 30
    ymin_detection_area_vehicles = int(img.shape[0]/100)*35
    xmax_detection_area_vehicles = int(img.shape[1])-30
    ymax_detection_area_vehicles = int(img.shape[0])-30
    
    # plot that area on new_img    
    if not cv2window and cv2imshow: cv2.rectangle(new_img, (xmin_detection_area_vehicles, ymin_detection_area_vehicles), (xmax_detection_area_vehicles, ymax_detection_area_vehicles), (0,150,150), 2)
    
            
    # if there are detections of vehicles
    if boxes_vehicles is not None: 
        # for each detection   
        for i in range(len(boxes_vehicles)):
            box_vehicle = boxes_vehicles[i]
            score_vehicle = scores_vehicles[i]

            # takes the coordinates of the bounding box of the vehicle detection
            xmin_vehicle, xmax_vehicle = (box_vehicle[0]).astype(int), (box_vehicle[2]).astype(int)
            ymin_vehicle, ymax_vehicle = (box_vehicle[1]).astype(int), (box_vehicle[3]).astype(int)
            
            
            # if the detection is outside of detection_area_vehicles, ignore them
            if(not cv2window and cv2imshow and ymin_vehicle < ymin_detection_area_vehicles): continue
            
            detections.append([xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle, score_vehicle])
                            
            # plot the detection on new_img
            cv2.rectangle(new_img, (xmin_vehicle, ymin_vehicle), (xmax_vehicle, ymax_vehicle), cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'])
            cv2.putText(new_img, f"vehicle {score_vehicle:.2f}", (xmin_vehicle, ymin_vehicle - 5), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])

            # show the img
            if plt_plot: 
                ax[0].imshow(cv2.cvtColor(img[ymin_vehicle:ymax_vehicle, xmin_vehicle:xmax_vehicle], cv2.COLOR_BGR2RGB))
                ax[0].set_title('img')   
                
            # define the box_plate_area as the area where is likely to find the plate
            box_plate_area = new_img.copy()   
            xmin_box_plate_area = xmin_vehicle + int((xmax_vehicle - xmin_vehicle)/4)
            xmax_box_plate_area = xmin_vehicle + int(((xmax_vehicle - xmin_vehicle)/4) * 3)
            ymin_box_plate_area = ymin_vehicle + int(((ymax_vehicle - ymin_vehicle)/10) * 4) 
            ymax_box_plate_area = ymax_vehicle
            
            # plot that area
            cv2.rectangle(box_plate_area, (xmin_box_plate_area, ymin_box_plate_area), (xmax_box_plate_area, ymax_box_plate_area), (0,255,225), -1)
            cv2.addWeighted(box_plate_area, 0.2, new_img, 1 - 0.2, 0, new_img)
                                                                                 
            # crop the image of detected vehicle
            vehicle_image = tensor_img.squeeze(dim=0)[:, ymin_vehicle:ymax_vehicle, xmin_vehicle:xmax_vehicle].unsqueeze(dim=0)  
                
            # recompute the model on the cropped image, to find plates
            predictions_plates = model(vehicle_image)[0]
                
            # decode predictions of plates
            box_plate, _, score_plate = decode_prediction_plates(prediction=predictions_plates, score_threshold=0.5)
            
            # if there are detections of plates
            if box_plate is not None:
                            
                # takes the coordinates of the bounding box of the vehicle detection
                xmin_plate, xmax_plate = (box_plate[0]).astype(int), (box_plate[2]).astype(int)
                ymin_plate, ymax_plate = (box_plate[1]).astype(int), (box_plate[3]).astype(int)
                
                # compute the overlap percentage between the detection of the plate and the box_plate_area. It should be at last of 80%
                overlap_percentage = compute_overlap_percentage((xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate, xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), 
                                                                (xmin_box_plate_area, ymin_box_plate_area, xmax_box_plate_area, ymax_box_plate_area))
                
                # compute the area percentage of plate box with reference the vehicle box. It should be between 2% and 4%
                area_percentage = compute_plate_area_percentage((xmin_plate, ymin_plate, xmax_plate, ymax_plate), 
                                                                (xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle))
                        
                if(overlap_percentage < 80 or (area_percentage < 2.0 or area_percentage > 4.0)): continue
                
                # crop the region of interest (roi), i.e. the plate
                roi = img[ymin_vehicle+ymin_plate:ymin_vehicle+ymax_plate, xmin_vehicle+xmin_plate:xmin_vehicle+xmax_plate]
                roi_to_save = roi;
                
                # if the super resolution weights are passed ar parameter, perform the super resolution.
                if sr_weights_path is not None:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(sr_weights_path)
                    sr.setModel("edsr",int(sr_weights_path.split('x')[1].split('.')[0]))
                    roi = sr.upsample(roi)
                
                # compute the gray image of the super resoluted roi
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                threshold = 160
                
                # compute how much pixels are under the threshold used to transfrom the gray image to a binary (b/w) image
                below_threshold_pixels = np.sum(gray_roi < threshold)
                percentage_below_threshold = (below_threshold_pixels / (gray_roi.shape[0]*gray_roi.shape[1])) * 100
                
                # if the pixels below the threshold are at least 50%      
                if(percentage_below_threshold > 50):
                    # compute the average distance of pixels below the threshold, from the threshold
                    below_threshold_mask = gray_roi < threshold
                    avg_distance2threshold = np.sum(threshold-gray_roi[below_threshold_mask])/(gray_roi.shape[0]*gray_roi.shape[1])
                    
                    # add to the gray image of roi the average distance to the threshold, plus 25. Then clip the gray roi
                    gray_roi = gray_roi + int(avg_distance2threshold)+25
                    gray_roi = np.clip(gray_roi, 0, 255).astype(np.uint8)

                # compute the binary roi
                _, binary_roi = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY)
                
                # compute the plate string
                plate_string, probability = string_plate_detector.detect_plate_string(plate_img=binary_roi, weights_dir='../weights/anpr_weights/')
                
                cv2.imwrite('./../plates/' + f'{plate_string}.jpeg', roi_to_save)
                
                # plot the plate 
                if plt_plot:                   
                    ax[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    ax[1].set_title('roi')
                    ax[2].imshow(gray_roi, cmap='gray')
                    ax[2].set_title('gray_roi')
                    ax[3].imshow(binary_roi, cmap='gray')
                    ax[3].set_title('binary_roi')     
                                                        
                # add plots to new_img
                cv2.rectangle(new_img, (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate), (xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), cv2_plates_cfg['color'], cv2_plates_cfg['thickness'])
                cv2.putText(new_img, f"plate {score_plate:.2f}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate - 5), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType']) 
                cv2.putText(new_img, f"{plate_string}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymax_plate + 30), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType'])    
                cv2.putText(new_img, f"{probability:.2f}", (xmin_vehicle+xmin_plate, ymin_vehicle+ymax_plate + 60), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], cv2_plates_cfg['color'], cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType'])    
    
    
    tracked_vehicles = {}
    if tracker is not None and len(detections)>0:
        tracks = tracker.update(np.array(detections))
        for i in range(len(tracks)):
            detection, track = detections[i], tracks[i]
            ymin, xmax, id = int(detection[1]), int(detection[2]), int(track[4])
            cv2.putText(new_img, f'id: {id}', (xmax-70, ymin-5), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])
            tracked_vehicles[f'{id}'] = [detection[0],detection[1],detection[2],detection[3]]
            
    if cv2imshow: cv2.imshow('Video', new_img)
                                                                    
    if cv2window: 
        while(1):
            if checkEsc(waiting_time=500): break
        cv2.destroyAllWindows()
            
    return new_img, tracked_vehicles
    



# VIDEO PROCESSING
def real_time_object_detector(model=None, video_path=None, sr_weights_path=None, velocity_cfg=None, cv2_vehicles_cfg=None, cv2_plates_cfg=None, new_frame_folder=None, frames_to_skip=-1):
        
    spd = StringPlateDetector()
        
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    
    line1 = velocity_cfg['line1']
    line2 = velocity_cfg['line2']
    meters_line1line2 = velocity_cfg['meters_line1line2']
    fps = velocity_cfg['fps']
    
    tracker = Sort()
    
    if new_frame_folder is not None: 
        if(not create_folder_if_not_exists(folder_path=new_frame_folder)):
            print('Error: please change the new_path_folder parameter (the chosen folder already exist.)')
            cv2.destroyAllWindows()
            cap.release()
            return   
        n=0

    old_tracker_vechiles = None
    velocity_tracked = {}
    
    while True:
        ret, frame = cap.read()
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES))<frames_to_skip: continue
        
        if ret == False:
            print('Error: Unable to read video.')
            break
              
        new_frame, new_tracked_vehicles = plot_one_image(model=model, img_param=frame, sr_weights_path=sr_weights_path, string_plate_detector=spd, tracker=tracker, cv2window=False, cv2imshow=True, plt_plot=False, cv2_vehicles_cfg=cv2_vehicles_cfg, cv2_plates_cfg=cv2_plates_cfg)     
                
        compute_velocity(line1=line1, line2=line2, meters_line1line2=meters_line1line2, fps=fps, new_tracked_vehicles=new_tracked_vehicles, old_tracker_vechiles=old_tracker_vechiles, velocity_tracked=velocity_tracked, new_frame=new_frame, cv2_vehicles_cfg=cv2_vehicles_cfg)
        cv2.imshow('Video', new_frame)               
        old_tracker_vechiles = new_tracked_vehicles
        
        if new_frame_folder is not None:   
            new_frame_name = compute_frame_name(n)
            if new_frame_name is None: 
                print('Error: frame number > 1000000000')
                break
            cv2.imwrite(new_frame_folder + f'{new_frame_name}.jpeg', new_frame)
            n+=1
        
        if checkEsc(waiting_time=500): break

    if new_frame_folder is not None:
        print('Creating new video ...')
        frames2video(images_folder=new_frame_folder, output_video_path=new_frame_folder,  original_video_path=video_path)
            
    cv2.destroyAllWindows()
    cap.release()
    return
    


############################################################################
# Internal use

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
    area_rect1 = np.abs(rect1[2] - rect1[0]) * np.abs(rect1[3] - rect1[1])
    area_rect2 = np.abs(rect2[2] - rect2[0]) * np.abs(rect2[3] - rect2[1])

    overlap_percentage = (area_intersection / min(area_rect1, area_rect2)) * 100

    return overlap_percentage


def compute_plate_area_percentage(plate, vehicle):
    xmin_plate, ymin_plate, xmax_plate, ymax_plate = plate[0], plate[1], plate[2], plate[3]
    xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle = vehicle[0], vehicle[1], vehicle[2], vehicle[3]
    
    width_plate = np.abs(xmax_plate-xmin_plate)
    height_plate = np.abs(ymax_plate-ymin_plate)
    width_vechile = np.abs(xmax_vehicle-xmin_vehicle)
    height_vehicle = np.abs(ymax_vehicle-ymin_vehicle)
    
    plate_area = width_plate*height_plate
    vehicle_area = width_vechile*height_vehicle
    area_percentage = (plate_area/vehicle_area)*100
    
    return area_percentage



def compute_velocity(line1=None, line2=None, meters_line1line2=None, fps=None, new_tracked_vehicles=None, old_tracker_vechiles=None, velocity_tracked=None, new_frame=None, cv2_vehicles_cfg=None):
    ymin_velocity_line = line1 if line1<line2 else line2
    ymax_velocity_line = line1 if line1>line2 else line2
    px2m = meters_line1line2/(np.abs(line2-line1))

    if old_tracker_vechiles is not None:
        for new_id, new_box in new_tracked_vehicles.items():
            if new_id not in velocity_tracked: velocity_tracked[f'{new_id}'] = []
            if new_box[3]<ymax_velocity_line and new_box[3]>ymin_velocity_line:
                for old_id, old_box in old_tracker_vechiles.items():
                    if old_id == new_id:      
                        # if the distance between the ymax of the same box in two consecutive frames is bigger than 30 px or smaller than 5 pixel -> skip
                        # if the mean between ymin and ymax of new box is higher than mean of ymin and ymax of old box -> skip
                        if (np.abs(new_box[3]-old_box[3])>30 or np.abs(new_box[3]-old_box[3])<5 or np.abs(new_box[3]-new_box[1]) > np.abs(old_box[3]-old_box[1])): continue 
                        
                        velocity_kmh = ((np.abs(old_box[3]-new_box[3])+np.abs(old_box[1]-new_box[1]))/2)*fps*px2m*3.6
                        velocity_tracked[f'{new_id}'].append(velocity_kmh)
                            
            elif new_box[3]<ymin_velocity_line and len(velocity_tracked[f'{new_id}'])>0:
                velocity_detected = np.mean(velocity_tracked[f'{new_id}'])
                cv2.putText(new_frame, f'Velocity: {velocity_detected:.2f} km/h', (new_box[0], new_box[3]+30), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])

    box_velocity_area = new_frame.copy()
    cv2.rectangle(box_velocity_area, (-5, ymin_velocity_line), (new_frame.shape[1]+5, ymax_velocity_line), (255,0,0), -1)
    cv2.addWeighted(box_velocity_area, 0.1, new_frame, 1 - 0.1, 0, new_frame)
    
def create_folder_if_not_exists(folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
        return True
    else:
        print(f"Folder '{folder_path}' already exist.")
        choice = input('Do you want save the new frame into it? (it not, please change the new_path_folder parameter) [y/n] ')
        while True:
            if choice.lower() == 'y':
                return True
            elif choice.lower() == 'n': 
                return False
            else:
                choice = input('Do you want save the new frame into it? (it not, please change the new_path_folder parameter) [y/n] ')
       
            
                
def compute_frame_name(n):
    if n<10: return f'000000000{n}'
    elif n>=10 and n<100: return f'00000000{n}'    
    elif n>=100 and n<1000: return f'0000000{n}'
    elif n>=1000 and n<10000: return f'000000{n}'
    elif n>=10000 and n<100000: return f'00000{n}'
    elif n>=100000 and n<1000000: return f'0000{n}'
    elif n>=1000000 and n<10000000: return f'000{n}'
    elif n>=10000000 and n<100000000: return f'00{n}'
    elif n>=100000000 and n<1000000000: return f'0{n}'
    elif n>=1000000000 and n<10000000000: return f'{n}'
    else: return None
        
        

def checkEsc(waiting_time=500):
    k = cv2.waitKey(waiting_time)
    if k==27: return True
    else: return False
        
        