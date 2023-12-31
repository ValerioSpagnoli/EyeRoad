import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

from glob import glob
import xml.etree.ElementTree as xet

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

size=(600,600)

# XML to CSV function to create dataset.csv file
def xml2csv_close_up(path_folder=None):
    
    path_xml_plates = sorted(glob(path_folder+"plates/*.xml"))
    path_xml_vehicles = sorted(glob(path_folder+"vehicles/*.xml"))
    
    labels_dict = dict(filepath=[],xmin=[],ymin=[],xmax=[],ymax=[],label=[])
    
    data = {'1':path_xml_vehicles, '2':path_xml_plates}
    for label, path in data.items():
        for filename in path:
            info = xet.parse(filename)
            root = info.getroot()
            member_object = root.find('object')
            labels_info = member_object.find('bndbox')
            xmin = int(labels_info.find('xmin').text)
            ymin = int(labels_info.find('ymin').text)
            xmax = int(labels_info.find('xmax').text)
            ymax = int(labels_info.find('ymax').text)

            splitted_filename = filename.split('/')
            filename_rel = './' + splitted_filename[-2] + '/' + splitted_filename[-1].split('.')[0] + '.jpeg'

            labels_dict['filepath'].append(filename_rel)
            labels_dict['xmin'].append(xmin)
            labels_dict['ymin'].append(ymin)
            labels_dict['xmax'].append(xmax)
            labels_dict['ymax'].append(ymax)
            
            labels_dict['label'].append(label)
        

    df = pd.DataFrame(labels_dict)
    df.to_csv(path_folder+'dataset.csv', index=False)
    return df


# XML to CSV function to create dataset.csv file
def xml2csv_traffic_camera(path_folder=None):
    
    path_xml = sorted(glob(path_folder+"*.xml"))
    labels_dict = dict(filepath=[],xmin=[],ymin=[],xmax=[],ymax=[],label=[])
    
    for filename in path_xml:
        info = xet.parse(filename)
        root = info.getroot()
        
        for i, member_object in enumerate(root.findall('object')):  
            obj1 = []
            obj2 = []
            name = member_object.find('name').text
            if name == 'vehicle':
                labels_info = member_object.find('bndbox')
                xmin = int(float(labels_info.find('xmin').text))
                ymin = int(float(labels_info.find('ymin').text))
                xmax = int(float(labels_info.find('xmax').text))
                ymax = int(float(labels_info.find('ymax').text))
                
                new_obj = [xmin, ymin, xmax, ymax]
                
                if np.abs(new_obj[2]-new_obj[0])*np.abs(new_obj[3]-new_obj[1]) < 1000: continue
                
                if len(obj1)!=4:
                    obj1 = new_obj
                elif len(obj2)!=4:
                    obj2 = new_obj
                else:
                    # if the new_obj is bigger than the obj1 -> the new_obj become the obj1 and the obj1 became the obj1
                    if np.abs(new_obj[2]-new_obj[0])*np.abs(new_obj[3]-new_obj[1]) > np.abs(obj1[2]-obj1[0])*np.abs(obj1[3]-obj1[1]): 
                        obj2 = obj1
                        obj1 = new_obj
                    # if the new_obj is lower than obj1 and greater than obj2 -> the new_obj become obj2
                    elif (np.abs(new_obj[2]-new_obj[0])*np.abs(new_obj[3]-new_obj[1]) < np.abs(obj1[2]-obj1[0])*np.abs(obj1[3]-obj1[1]) and 
                            np.abs(new_obj[2]-new_obj[0])*np.abs(new_obj[3]-new_obj[1]) > np.abs(obj2[2]-obj2[0])*np.abs(obj2[3]-obj2[1])): 
                        obj2 = new_obj
            
        if len(obj1) == 4:
            splitted_filename = filename.split('/')
            filename_rel = splitted_filename[-1].split('.')[0] + '.jpeg'
            labels_dict['filepath'].append(filename_rel)
            labels_dict['xmin'].append(obj1[0])
            labels_dict['ymin'].append(obj1[1])
            labels_dict['xmax'].append(obj1[2])
            labels_dict['ymax'].append(obj1[3])
            labels_dict['label'].append('1')
            
        if len(obj2) == 4:
            splitted_filename = filename.split('/')
            filename_rel = splitted_filename[-1].split('.')[0] + '.jpeg'
            labels_dict['filepath'].append(filename_rel)
            labels_dict['xmin'].append(obj1[0])
            labels_dict['ymin'].append(obj1[1])
            labels_dict['xmax'].append(obj1[2])
            labels_dict['ymax'].append(obj1[3])
            labels_dict['label'].append('1')
        
    df = pd.DataFrame(labels_dict)
    df.to_csv(path_folder+'dataset.csv', index=False)
    return df



# Return the transfomation will be applied on datasets
def get_transform(train):
    transform = []

    transform.append(transforms.ToTensor())
    transform.append(transforms.Resize(size))
    
    if train == True:
        transform.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        transform.append(transforms.RandomErasing())
        transform.append(transforms.RandomGrayscale())

    return transforms.Compose(transform)


# Class of dataset
# Return image, dict{'boxes': [xmin, ymin, xmax, ymax], 'label':[1], 'image_id':000}
class PlateDetectorDataset(Dataset):
    def __init__(self, data_dir=None, csv_file=None, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir+csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = self.data_dir+self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
                
        xmin, ymin, xmax, ymax, label = self.data.iloc[idx, 1:].values
        image_id = int(img_path.split('/')[-1].split('.')[0])
                
        scale_factor = (size[0]/image.size[0], size[1]/image.size[1])

        box = [xmin, ymin, xmax, ymax]
        scaled_box = box
        scaled_box[0] *= scale_factor[0]  # x_min 
        scaled_box[2] *= scale_factor[0]  # x_max
        scaled_box[1] *= scale_factor[1]  # y_min
        scaled_box[3] *= scale_factor[1]  # y_max

        target = {}
        target["boxes"] = torch.as_tensor([scaled_box], dtype = torch.float32)
        target["labels"] = torch.as_tensor([label], dtype = torch.int64)
        target["image_id"] = torch.as_tensor(image_id)
        
        if self.transform:
            image = self.transform(image)

        return image, target


# Return splitted dataset
def get_datasets(data_dir=None, csv_file=None):
    train_ds = PlateDetectorDataset(data_dir=data_dir, csv_file=csv_file, transform=get_transform(train=True))
    valid_ds = PlateDetectorDataset(data_dir=data_dir, csv_file=csv_file, transform=get_transform(train=False))

    indices = torch.randperm(len(train_ds)).tolist()
    train_indices = indices[:int(len(indices) * 0.8):]
    valid_indices = indices[int(len(indices) * 0.8):]
        
    # Train dataset: 80% of the entire data
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    
    # Validation dataset: 20% of the entire data
    valid_ds = torch.utils.data.Subset(valid_ds, valid_indices)

    return train_ds, valid_ds


def dataset_stats(dataset=None):
    dataset_counts = {'1':0, '2':0}
    for i in range(dataset.__len__()):
        _, target = dataset.__getitem__(i)
        label = target['labels'].item()
        dataset_counts[str(label)]+=1

    print('Number of images of class 1: {}'.format(dataset_counts['1']))
    print('Number of images of class 2: {}'.format(dataset_counts['2']))


# Plot images with ground truth bounding box
def plot_image_from_dataset(dataset=None, idx=None):
    
    img, target = dataset.__getitem__(idx)
    box = target['boxes'].cpu().detach().numpy()[0]

    # denormalize image
    img = img.numpy().transpose((1, 2, 0))

    # denormalize labels
    xmin, xmax = (box[0]).astype(int), (box[2]).astype(int)
    ymin, ymax = (box[1]).astype(int), (box[3]).astype(int)

    fig, ax = plt.subplots()
    ax.imshow(img)

    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()
    


# Return dataloader
def get_dataloaders(train_ds=None, valid_ds=None, batch_size=8):
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: list(zip(*batch)))
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: list(zip(*batch)))
    
    return (train_dl, valid_dl)



    
    

    


