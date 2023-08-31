import cv2
import os

def frames2video(images_folder=None, output_video_path=None, original_video_path=None, fps=25):
    
    print('Creating new video ...')
    
    if original_video_path is not None:
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f'FPS detected from original video: {fps}')

    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpeg')]
    image_files.sort()

    first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
    height, width, _ = first_image.shape
 
    video_writer = cv2.VideoWriter(output_video_path+'output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    print("Video created successfully!")