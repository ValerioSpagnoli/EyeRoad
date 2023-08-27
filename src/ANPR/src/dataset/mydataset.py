import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import string


# Build dataset from a csv file.
class LicensePlatesDataset(data.Dataset):
    # The csv file should have the following header: (img_path, label, split)

    def __init__(self, root_dir, csv_file, max_len, split, transform=None):
        super(LicensePlatesDataset, self).__init__()
        # Define root dir
        self.root_dir = root_dir
        # Build data frame
        df = pd.read_csv(csv_file)
        self.data =  df[df['split'] == split]
        # Define transformation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Define image path
        img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
        # Open image with PIL
        image = Image.open(img_path)
        # Define plate id (plate number)
        plate_id = self.data.iloc[index, 1]

        # Define label length
        label_len = torch.tensor(len(plate_id)+1)

        # Apply transformations (if needed)
        if self.transform:
            image = self.transform(image)

        # Return image, label, and label_len
        return image, plate_id, label_len
