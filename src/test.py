from modules import inference, real_time_object_detector

detector = inference.load_model_for_inference(weights_file='weights/detector_weights/detector.pt', device='cpu')
velocity_cfg = {'line1': 1025, 'line2': 1205, 'meters_line1line2':7.5, 'fps':25}

real_time_object_detector.real_time_object_detector(model=detector, 
                                                    video_path='video_test/video_test.mp4', 
                                                    sr_weights_path='weights/edsr_weights/EDSR_x3.pb',
                                                    spl_weights_path='weights/alpr_weights/ALPR_weights.pth.zip',
                                                    velocity_cfg=velocity_cfg,
                                                    new_frame_folder='output_video/',
                                                    frames_to_skip=0)
