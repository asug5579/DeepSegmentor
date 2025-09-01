# Author: Yahui Liu <yahui.liu@unitn.it>
# Editor: edits made by asug5579 to suit the GAPs384 dataset
# Cite: Claude, GPT5, Gemini2.5 assisted in below class setup and writing code for Yolov11 conversion

import os.path
import cv2
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
import torch
from data.image_folder import make_dataset

class GAPs384Dataset(BaseDataset):
    
    def __init__(self, opt):
        """
        opt: options object containing dataset configuration
        """
        BaseDataset.__init__(self, opt)
        
        # load all images (test, train, valid, etc all in same place)
        all_img_paths = make_dataset(os.path.join(opt.dataroot, 'train/images'))
        
        # filter out what is needed (whether training, testing or validating)
        if opt.phase == 'train':
            self.img_paths = [p for p in all_img_paths if os.path.basename(p).startswith('train_')]
            print(f"Training phase: using {len(self.img_paths)} images")
        elif opt.phase == 'val':
            self.img_paths = [p for p in all_img_paths if os.path.basename(p).startswith('valid_')]
            print(f"Validation phase: using {len(self.img_paths)} images")
        elif opt.phase == 'test':
            self.img_paths = [p for p in all_img_paths if os.path.basename(p).startswith('test_')]
            print(f"Testing phase: using {len(self.img_paths)} images")
        
        # where labels are stored
        self.lab_dir = os.path.join(opt.dataroot, 'train/labels')

        # image transformations: convert to tensor and normalize to [-1, 1]
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize range
        ])
        
    def yolo11_to_binary_mask(self, yolo_coords, img_width, img_height):
        """
        Convert YOLOv11 polygon coordinates to binary mask.
            
        Returns:
            Binary mask where cracks are 255 (will be converted to 1 later)
        """
        # create empty mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # set coordinates to pixels
        coords = []
        for i in range(0, len(yolo_coords), 2):
            if i + 1 < len(yolo_coords):
                x = float(yolo_coords[i]) * img_width
                y = float(yolo_coords[i + 1]) * img_height
                coords.append([x, y])
        
        # fill polygon with white (255)
        if len(coords) >= 3:
            coords = np.array(coords, dtype=np.int32)
            cv2.fillPoly(mask, [coords], 255)
        
        return mask
        
    def __getitem__(self, index):
        """
        Load, process and return single image-label pair.
        """
        
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

        # getcorresponding label
        img_name = os.path.basename(img_path).rstrip('.jpg')
        lab_path = os.path.join(self.lab_dir, img_name + '.txt')
        #print(lab_path)

        # create empty mask
        lab = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # convert yolo dataset into binary form
        if os.path.exists(lab_path):
            with open(lab_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 2:
                        #class_id = int(parts[0])
                        coords = parts[1:]
                        # convert this polygon to mask and combine with existing mask
                        polygon_mask = self.yolo11_to_binary_mask(coords, img.shape[1], img.shape[0])
                        lab = cv2.bitwise_or(lab, polygon_mask)  # combine masks using OR
        else:
            print(f"Warning: No label file found for {img_path}")

        # resize image and labels
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        lab = cv2.resize(lab, (400, 400), interpolation=cv2.INTER_NEAREST)

        # convert label to binary (0 or 1) and ensure float32
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        lab = lab.astype(np.float32)

        # convert to PyTorch tensors
        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = torch.from_numpy(lab).float().unsqueeze(0)

        return {
            'image': img, # Input image tensor (3, 40, 40) in range [-1, 1]
            'label': lab, # Binary mask tensor (1, 40, 40) in range [0, 1] as FLOAT
            'A_paths': img_path, # Path to input image
            'B_paths': lab_path # Path to label file
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths) 