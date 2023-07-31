# Libraries
import torch
import platform
import sys
import cv2
from modules import dataset, training, model, inference, real_time_object_detector


def main(prepare_csv=False, train_model=False, load_model=False, compute_one_image=False, compute_real_time_object_detector=False):

    if len(sys.argv) != 6:
        print('Usage: python main.py prepare_scv=<True,False>')
        print('                      train_model=<True,False>')
        print('                      load_model=<True,False>')
        print('                      compute_one_image=<True,False>')
        print('                      compute_real_time_object_detector=<True,False>')
        return

    PREPARE_CSV = sys.argv[1]=='True'
    TRAIN_MODEL = sys.argv[2]=='True'
    LOAD_MODEL = sys.argv[3]=='True'
    COMPUTE_ONE_IMAGE = sys.argv[4]=='True'
    COMPUTE_REAL_TIME_OBJECT_DETECTOR = sys.argv[5]=='True'
    
    
    print('Main ... ')
    print('Parameters:')
    print(f' - PREPARE_CSV = {PREPARE_CSV}')
    print(f' - TRAIN_MODEL = {TRAIN_MODEL}')
    print(f' - LOAD_MODEL = {LOAD_MODEL}')
    print(f' - COMPUTE_ONE_IMAGE = {COMPUTE_ONE_IMAGE}')
    print(f' - COMPUTE_REAL_TIME_OBJECT_DETECTOR = {COMPUTE_REAL_TIME_OBJECT_DETECTOR}')
    print('-------------------------------------------------------------------------------------------------')
    
    ########################################################################################################################################
    
    
    if PREPARE_CSV: 
        print('Preparing csv ...')
        _ = dataset.xml2csv(main_dir=main_dir, path_folder=data_dir)
        print('... done!\n')
    
    print('-------------------------------------------------------------------------------------------------')    
    
    print('Getting datasets and dataloaders ...')
    train_ds, valid_ds = dataset.get_datasets(data_dir=main_dir+data_dir, csv_file='dataset.csv')
    train_dl, valid_dl = dataset.get_dataloaders(train_ds=train_ds, valid_ds=valid_ds, batch_size=8)
    print('... done!\n')
    print('-------------------------------------------------------------------------------------------------')    
    
    
    ########################################################################################################################################
    
    
    if TRAIN_MODEL:
        print('Download pretrained model ...')
        detector = model.get_plateDetectorModel(num_classes=3, feature_extraction=True)
        print('... done!\n')
        if LOAD_MODEL:
            print('Loading model for training ...')
            detector.load_state_dict(torch.load(main_dir+weights_dir+'detector_weights/detector.pt'))
            print('... done!\n')

        optimizer = torch.optim.SGD([p for p in detector.parameters() if p.requires_grad],
                                    lr = 0.005,
                                    momentum = 0.9,
                                    weight_decay = 0.0005)
        print('Training ...')
        losses = training.training_and_validation(model=detector.to(device),
                                                    optimizer=optimizer,
                                                    num_epochs=50,
                                                    train_loader=train_dl,
                                                    valid_loader=valid_dl,
                                                    device=device,
                                                    weights_dir=main_dir+weights_dir+'detector_weights/',
                                                    verbose=2)
        print('... done!\n')
        print('-------------------------------------------------------------------------------------------------')
        
        training.plot_losses(train_epoch_losses=losses[0], valid_epoch_losses=losses[1])
        
    
    ########################################################################################################################################
    
    
    if COMPUTE_ONE_IMAGE: 
        print('Compute one image ...')
        detector = inference.load_model_for_inference(weights_file=main_dir+weights_dir+'detector_weights/detector.pt', device=device)
        cv2_vehicles_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(0,0,255), 'thickness':2, 'lineType':cv2.LINE_AA}
        cv2_plates_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(255,0,0), 'thickness':2, 'lineType':cv2.LINE_AA}
        real_time_object_detector.plot_one_image(model=detector, 
                                                img_param=data_dir+'test/TEST_2.jpeg', 
                                                sr_weights_path=weights_dir+'edsr_weights/EDSR_x3.pb', 
                                                cv2window=True, 
                                                cv2imshow=True, 
                                                plt_plot=False, 
                                                cv2_vehicles_cfg=cv2_vehicles_cfg, 
                                                cv2_plates_cfg=cv2_plates_cfg)
        print('... done!\n')
        print('-------------------------------------------------------------------------------------------------')
        
        
    ########################################################################################################################################    
        
        
    if COMPUTE_REAL_TIME_OBJECT_DETECTOR: 
        print('Compute real time object detector ...')
        detector = inference.load_model_for_inference(weights_file=main_dir+weights_dir+'detector_weights/detector.pt', device=device)
        cv2_vehicles_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(0,0,255), 'thickness':2, 'lineType':cv2.LINE_AA}
        cv2_plates_cfg = {'fontFace':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.7, 'color':(255,0,0), 'thickness':2, 'lineType':cv2.LINE_AA}
        real_time_object_detector.real_time_object_detector(model=detector, 
                                                            video_path=data_dir+'test/TEST.mp4', 
                                                            sr_weights_path=weights_dir+'edsr_weights/EDSR_x3.pb',
                                                            cv2_vehicles_cfg=cv2_vehicles_cfg, 
                                                            cv2_plates_cfg=cv2_plates_cfg,
                                                            new_frame_folder=main_dir+'../output_video/')
        print('... done!\n')
        print('-------------------------------------------------------------------------------------------------')
    
    
    return    
    
    
if __name__ == "__main__":
    
    print('Detecting system ...')
    
    # MacOs
    if platform.system() == "Darwin":
        main_dir = './'
        data_dir = '../data/'
        weights_dir = '../weights/'
        
        device = 'cpu'
        #device = 'mps' if torch.backends.mps.is_available()  else 'cpu'
        
        print(f"System: MacOs\nDevice: {device}\nMain_dir: {main_dir}")

    # Linux
    elif platform.system() == "Linux":
        main_dir = './'
        data_dir = '../data/'
        detector_weights_dir = '../weights/'
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"System: Linux\nDevice: {device}\nMain_dir: {main_dir}")

    # Unknown
    else:
        print("System unknown")
    
    print('... done!\n')
    print('-------------------------------------------------------------------------------------------------')
        
        
    main()