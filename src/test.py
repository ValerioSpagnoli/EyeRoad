from modules import inference, real_time_object_detector
from utils import frames2video

detector = inference.load_model_for_inference(weights_file='weights/detector_weights/detector.pt', device='cpu')
velocity_cfg = {'line1': 1025, 'line2': 1205, 'meters_line1line2':7.5, 'fps':25}

real_time_object_detector.real_time_object_detector(model=detector, 
                                                    video_path='video_test/video_test.mp4', 
                                                    sr_weights_path='weights/edsr_weights/EDSR_x3.pb',
                                                    spl_weights_path='weights/alpr_weights/ALPR_weights.pth.zip',
                                                    velocity_cfg=velocity_cfg,
                                                    new_frame_folder='output_video/',
                                                    start_frame=0,
                                                    end_frame=4)

frames2video.frames2video(images_folder='output_video/', 
                          output_video_path='output_video/', 
                          original_video_path='video_test/video_test.mp4', 
                          fps=25)


