import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from difflib import SequenceMatcher

from modules.inference import decode_prediction_vehicles, decode_prediction_plates
from modules.detect_plate_string import StringPlateDetector
from utils.frames2video import frames2video


# IMAGE PROCESSING
def plot_one_image(model=None, image=None, sr_weights_path=None, string_plate_detector=None, ids=None, cv2_vehicles_cfg=None, cv2_plates_cfg=None):
            
    detections={}        
            
    # if configurations of plot are not passed as paramenters, set them
    if cv2_vehicles_cfg is None: cv2_vehicles_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'color':(0,0,255), 'thickness':2, 'lineType':cv2.LINE_AA}
    if cv2_plates_cfg is None: cv2_plates_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'color':(255,0,0), 'thickness':2, 'lineType':cv2.LINE_AA}
        
    # check if the image passed as parameter is an cv2 image, or a string (i.e. the path of the image)    
    if(isinstance(image, np.ndarray)): img = image
    else: print('Error: unable to read image.')
    
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
            if(ymin_vehicle < ymin_detection_area_vehicles): continue
                                                                       
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
                
                # compute plate area and discard all detections under 1500 px^2 and over 10000 px^2 of area
                plate_area = np.abs(xmax_plate-xmin_plate)*np.abs(ymax_plate-ymin_plate)
                if plate_area < 1500 or plate_area > 10000: continue

                # crop the region of interest (roi), i.e. the plate
                roi = img[ymin_vehicle+ymin_plate:ymin_vehicle+ymax_plate, xmin_vehicle+xmin_plate:xmin_vehicle+xmax_plate]
                
                # if the super resolution weights are passed ar parameter, perform the super resolution.
                if sr_weights_path is not None:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(sr_weights_path)
                    sr.setModel("edsr",int(sr_weights_path.split('x')[1].split('.')[0]))
                    sr_roi = sr.upsample(roi)

                # compute the gray image of the super resoluted roi
                gray_roi = cv2.cvtColor(sr_roi, cv2.COLOR_BGR2GRAY)

                # compute the plate string
                plate_string = string_plate_detector.detect_plate_string(image=gray_roi)
                                                                
                # add plots to new_img
                cv2.rectangle(new_img, (xmin_detection_area_vehicles, ymin_detection_area_vehicles), (xmax_detection_area_vehicles, ymax_detection_area_vehicles), (0,150,150), 2)
                cv2.rectangle(new_img, (xmin_vehicle+xmin_plate, ymin_vehicle+ymin_plate), (xmin_vehicle+xmax_plate, ymin_vehicle+ymax_plate), cv2_plates_cfg['color'], cv2_plates_cfg['thickness'])
                cv2.rectangle(new_img, (xmin_vehicle, ymin_vehicle), (xmax_vehicle, ymax_vehicle), cv2_vehicles_cfg['color'], cv2_vehicles_cfg['thickness'])
                
                cv2.rectangle(new_img, (xmin_vehicle, ymin_vehicle), (xmax_vehicle, ymin_vehicle+25), cv2_vehicles_cfg['color'], -1)
                cv2.putText(new_img, f"vehicle: {score_vehicle:.2f}", (xmin_vehicle+5, ymin_vehicle+20), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], (255,255,255), cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])
                
                plate_background = new_img.copy()
                cv2.rectangle(plate_background, (xmin_vehicle+xmin_plate-2, ymin_vehicle+ymin_plate-25), (xmin_vehicle+xmax_plate+2, ymin_vehicle+ymin_plate), cv2_plates_cfg['color'], -1)
                cv2.rectangle(plate_background, (xmin_vehicle+xmin_plate-2, ymin_vehicle+ymax_plate), (xmin_vehicle+xmax_plate+2, ymin_vehicle+ymax_plate+25), cv2_plates_cfg['color'], -1)
                cv2.addWeighted(plate_background, 0.6, new_img, 1 - 0.6, 0, new_img)
                cv2.putText(new_img, f"plate {score_plate:.2f}", (xmin_vehicle+xmin_plate+5, ymin_vehicle+ymin_plate - 5), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], (255,255,255), cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType']) 
                cv2.putText(new_img, f"{plate_string}", (xmin_vehicle+xmin_plate+5, ymin_vehicle+ymax_plate + 20), cv2_plates_cfg['fontFace'], cv2_plates_cfg['fontScale'], (255,255,255), cv2_plates_cfg['thickness'], cv2_plates_cfg['lineType'])        
                
                compute_id(ids=ids, detections=detections, plate_string=plate_string, box_vehicle=[xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle], frame=new_img, cv2_vehicle_cfg=cv2_vehicles_cfg)
                                                                                                                    
    return new_img, detections
    



# VIDEO PROCESSING
def real_time_object_detector(model=None, video_path=None, sr_weights_path=None, spl_weights_path=None, velocity_cfg=None, cv2_vehicles_cfg=None, cv2_plates_cfg=None, new_frame_folder=None, frames_to_skip=-1):
        
    # define open cv window
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    
    # define string plate detector object
    spd = StringPlateDetector(weights_path=spl_weights_path)
    
    # create folder for frames of video and output video    
    if new_frame_folder is not None: 
        if(not create_folder_if_not_exists(folder_path=new_frame_folder)):
            print('Error: please change the new_path_folder parameter (the chosen folder already exist.)')
            cv2.destroyAllWindows()
            cap.release()
            return   
        n=0
    
    
    old_tracked_vehicles = None # key='id', value=box    
    ids = {} # key=plate_sting, value=id
    n_nontracked_frames = {}
    velocity_tracked = {}
    
    while True:
        ret, frame = cap.read()
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES))<frames_to_skip: continue
        
        if ret == False:
            print('Error: Unable to read video.')
            break
              
        new_frame, new_tracked_vehicles = plot_one_image(model=model, image=frame, sr_weights_path=sr_weights_path, string_plate_detector=spd, 
                                                         ids=ids, cv2_vehicles_cfg=cv2_vehicles_cfg, cv2_plates_cfg=cv2_plates_cfg)    
        
        
        i = 0
        keys = list(ids.keys())
        values = list(ids.values())
        while i<len(keys):
            key = keys[i]
            value = values[i]
            if value not in new_tracked_vehicles.keys():
                if value not in n_nontracked_frames.keys():
                    n_nontracked_frames[value] = 0
                else:
                    if n_nontracked_frames[value]>2:
                        ids.pop(key)
                        n_nontracked_frames.pop(value)
                    else:
                        n_nontracked_frames[value] = n_nontracked_frames[value]+1
            else:
                if value in n_nontracked_frames.keys():
                    n_nontracked_frames[value] = 0
            i=i+1
                 
                
        compute_velocity(line1=velocity_cfg['line1'], line2=velocity_cfg['line2'], meters_line1line2=velocity_cfg['meters_line1line2'], 
                         fps=velocity_cfg['fps'], new_tracked_vehicles=new_tracked_vehicles, old_tracked_vechiles=old_tracked_vehicles, 
                         velocity_tracked=velocity_tracked, new_frame=new_frame, cv2_vehicles_cfg=cv2_vehicles_cfg)
        
        old_tracked_vehicles = new_tracked_vehicles

        if new_frame_folder is not None:   
            new_frame_name = compute_frame_name(n)
            if new_frame_name is None: 
                print('Error: frame number > 1000000000')
                break
            cv2.imwrite(new_frame_folder + f'{new_frame_name}.jpeg', new_frame)
            n+=1
            
        cv2.imshow('Video', new_frame)
        
        if checkEsc(waiting_time=500): break

    # create video from frames
    if new_frame_folder is not None: frames2video(images_folder=new_frame_folder, output_video_path=new_frame_folder,  original_video_path=video_path)
            
    cv2.destroyAllWindows()
    cap.release()
    return
    


############################################################################
# Internal use

def compute_id(ids, detections, plate_string, box_vehicle, frame, cv2_vehicle_cfg):
    
    xmin_vehicle = box_vehicle[0]
    ymin_vehicle = box_vehicle[1]
    xmax_vehicle = box_vehicle[2]
    ymax_vehicle = box_vehicle[3]
    
    if ids.get(plate_string) is None:
        found = False
        for key in ids.keys():
            if SequenceMatcher(None, key, plate_string).ratio() > 0.8:
                plate_string = key
                found = True
                break
            
        if not found:
            frist_id_free = 0
            new_id_found = False
            while not new_id_found:
                values = ids.values()
                if frist_id_free in values:
                    frist_id_free = frist_id_free+1
                else:
                    new_id_found = True
            ids[plate_string] = frist_id_free
            
    cv2.putText(frame, f"id: {ids[plate_string]}", (xmin_vehicle+170, ymin_vehicle + 20), cv2_vehicle_cfg['fontFace'], cv2_vehicle_cfg['fontScale'], (255,255,255), cv2_vehicle_cfg['thickness'], cv2_vehicle_cfg['lineType'])        
    detections[ids[plate_string]] = [xmin_vehicle, ymin_vehicle, xmax_vehicle, ymax_vehicle]
    return detections



def compute_velocity(line1=None, line2=None, meters_line1line2=None, fps=None, new_tracked_vehicles=None, old_tracked_vechiles=None, velocity_tracked=None, new_frame=None, cv2_vehicles_cfg=None):
    ymin_velocity_line = line1 if line1<line2 else line2
    ymax_velocity_line = line1 if line1>line2 else line2
    px2m = meters_line1line2/(np.abs(line2-line1))

    if old_tracked_vechiles is not None:
        for new_id, new_box in new_tracked_vehicles.items():
            if new_id not in velocity_tracked: 
                velocity_tracked[new_id] = []
            if new_box[3]<ymax_velocity_line and new_box[3]>ymin_velocity_line:
                for old_id, old_box in old_tracked_vechiles.items():
                    if old_id == new_id:      
                        # if the distance between the ymax of the same box in two consecutive frames is bigger than 30 px or smaller than 5 pixel -> skip
                        # if the mean between ymin and ymax of new box is higher than mean of ymin and ymax of old box -> skip
                        if (np.abs(new_box[3]-old_box[3])>30 or np.abs(new_box[3]-old_box[3])<5 or np.abs(new_box[3]-new_box[1]) > np.abs(old_box[3]-old_box[1])): continue 
                        velocity_kmh = ((np.abs(old_box[3]-new_box[3])+np.abs(old_box[1]-new_box[1]))/2)*fps*px2m*3.6
                        velocity_tracked[new_id].append(velocity_kmh)
                            
            elif new_box[3]<ymin_velocity_line and len(velocity_tracked[new_id])>0:
                velocity_detected = np.mean(velocity_tracked[new_id])
                cv2.putText(new_frame, f'{velocity_detected:.2f} km/h', (new_box[0]+250, new_box[1]+20), cv2_vehicles_cfg['fontFace'], cv2_vehicles_cfg['fontScale'], (255,255,255), cv2_vehicles_cfg['thickness'], cv2_vehicles_cfg['lineType'])

    box_velocity_area = new_frame.copy()
    cv2.rectangle(box_velocity_area, (-5, ymin_velocity_line), (new_frame.shape[1]+5, ymax_velocity_line), (255,0,0), -1)
    cv2.addWeighted(box_velocity_area, 0.05, new_frame, 1 - 0.05, 0, new_frame)
    
    
    
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
        
        