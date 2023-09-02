import cv2
import os
import imageio
import shutil
 
def video2gif(folder=None, relative_video_path=None, scale_factor=1.0, compression_factor=100):
    
    # check if video exists.
    if folder == None:
        print('Error: folder is None.')
        return
    
    if relative_video_path == None:
        print('Error: relative video path is None.')
    else:
        if not os.path.exists(os.path.join(folder, relative_video_path)):
            print(f'Error: cannot find {relative_video_path} into {folder}.')
            return


    # create temp folder
    if not os.path.exists(f'{folder}/temp'):
        try:
            os.mkdir(f'{folder}/temp')
            print(f'Created folder {folder}/temp')
        except OSError as e:
            print(f'Error creating folder {folder}/temp: {e}')
            return
    else:
        try:
            shutil.rmtree(f'{folder}/temp')
            print(f'Deleted folder {folder}/temp.')
            try:
                os.mkdir(f'{folder}/temp')
                print(f'Created folder {folder}/temp.')
            except OSError as e:
                print(f'Error creating folder {folder}/temp: {e}')
                return
            
        except OSError as e:
            print(f'Error deleting folder {folder}/temp: {e}')
            return


    print(f'Start processing {relative_video_path} ...')

    cap = cv2.VideoCapture(f'{folder}/{relative_video_path}')
    if not cap.isOpened():
        print(f'Error: cannot open {folder}/{relative_video_path}')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    i=0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.resize(frame, (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor)))
        
        if i<10:
            frame_name = f'00{i}.jpg'
        elif i>=10 and i<100:
            frame_name = f'0{i}.jpg'
        elif i>=100:
            frame_name = f'{i}.jpg'
        
        cv2.imwrite(f'{folder}/temp/{frame_name}', frame, [cv2.IMWRITE_JPEG_QUALITY, compression_factor])
        print(f'Processing frame {i}')
        i=i+1
    cap.release()
    print('done!')

    image_files = sorted([f for f in os.listdir(f'{folder}/temp/') if f.endswith(('.jpg'))])
    print(f'Processed {len(image_files)} frames.')
    print('Creating gif ...')
    
    images = []
    for image_file in image_files:
        img_path = f'{folder}/temp/{image_file}'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to open {img_path}")

    # Convert to gif using the imageio.mimsave method
    gif_name = relative_video_path.split('.')[0]
    imageio.mimsave(f'{folder}/{gif_name}.gif', images, duration=int((1/fps)*1000))
    
    print('done!')
    
if __name__ == "__main__":
    video2gif(folder='media', relative_video_path='video_demo.mp4', scale_factor=0.30, compression_factor=85)