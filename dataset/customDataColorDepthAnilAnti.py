'''
For CelebA-Spoof
'''
from torch.utils.data import Dataset
from imutils import paths
from PIL import Image, ImageOps
import numpy as np
import torch
import random
import os

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_rgb_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def default_color_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    # RGBimg = RGBimg.resize((380,380))
    # HSVimg = HSVimg.resize((380,380))
    RGBimg = RGBimg.resize((224,224))
    HSVimg = HSVimg.resize((224,224))
    return RGBimg, HSVimg

def default_depth_loader(path,imgsize=112):
    img = Image.open(path)
    re_img = img.resize((imgsize,imgsize), resample = Image.BICUBIC)
    re_img = ImageOps.grayscale(re_img)
    return re_img

def load_mat(path):
    mat = np.loadtxt(path)
    return torch.tensor(mat,dtype=torch.float32)

class customData(Dataset):
    def __init__(self, img_path, txt_path, phase = '',data_transforms_color=None, data_transforms_depth=None, color_loader = default_color_loader, depth_loader = default_depth_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.Random(4).shuffle(lines)
            # self.img_name = [os.path.join(img_path, (line.strip().split(' ')[0][11:])) for line in lines]
            self.img_name = []
            self.depth_name = []
            self.img_label = []
            lengthOfTrain = int(len(lines)*0.8)
            endOfVal = int(len(lines))#*0.9)
            if phase=='meta_train':
                for line in lines[:lengthOfTrain]:
                    rgb_path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    depth_path = rgb_path.replace("trainSquareCropped", "trainSquareCropped_depth")
                    label = int(line.strip().split(' ')[-1])
                    if os.path.exists(depth_path):
                        self.img_name.append(rgb_path)
                        self.depth_name.append(depth_path)
                        self.img_label.append(label)
                    else:
                        print(rgb_path)
            elif phase=='meta_val':
                for line in lines[lengthOfTrain:endOfVal]:
                    rgb_path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    depth_path = rgb_path.replace("trainSquareCropped", "trainSquareCropped_depth")
                    label = int(line.strip().split(' ')[-1])
                    if os.path.exists(depth_path):
                        self.img_name.append(rgb_path)
                        self.depth_name.append(depth_path)
                        self.img_label.append(label)
                    else:
                        print(rgb_path)

            elif phase=='meta_test':
                for line in lines[endOfVal:]:
                    rgb_path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    depth_path = rgb_path.replace("trainSquareCropped", "trainSquareCropped_depth")
                    label = int(line.strip().split(' ')[-1])
                    if os.path.exists(depth_path):
                        self.img_name.append(rgb_path)
                        self.depth_name.append(depth_path)
                        self.img_label.append(label)
                    else:
                        print(rgb_path)
                        
            elif phase=='test':
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][10:]))
                    if os.path.isfile(path):
                        if "live" in path:# and getreal:
                            label = 0
                        else:#if not getreal:
                            label = 1
                        self.img_name.append(path)
                        self.img_label.append(label)

            elif phase=="ssl":
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][11:]))
                    label = int(line.strip().split(' ')[-1])
                    if os.path.isfile(path):
                        self.img_name.append(path)
                        self.img_label.append(label)
                    else:
                        print(path)

            # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms_color = data_transforms_color
        self.data_transforms_depth = data_transforms_depth
        self.phase = phase
        self.color_loader = color_loader
        self.depth_loader = depth_loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            depth_name = self.depth_name[item]
            label = self.img_label[item]
            rgb_img, hsv_img = self.color_loader(img_name)
            depth_img = self.depth_loader(depth_name)

            if self.data_transforms_color is not None:
                try:
                    rgb_img = self.data_transforms_color(rgb_img)
                    hsv_img = self.data_transforms_color(hsv_img)
                    depth_img = self.data_transforms_depth(depth_img)
                    color_img = torch.cat([rgb_img, hsv_img], 0)
                    img = [color_img, depth_img]
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, label
            # return rgb_img, depth_img, label
        except:
            print(item)

class customTargetData(Dataset):
    def __init__(self, img_path, txt_path, phase = '',data_transforms_color=None, data_transforms_depth=None, color_loader = default_color_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.Random(4).shuffle(lines)
            self.img_name = []
            self.img_label = []
    
            if phase=='test':
                for line in lines:
                    path = os.path.join(img_path, (line.strip().split(' ')[0][10:]))
                    if os.path.isfile(path):
                        if "live" in path:# and getreal:
                            label = 0
                        else:#if not getreal:
                            label = 1
                        self.img_name.append(path)
                        self.img_label.append(label)

            # self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms_color = data_transforms_color
        self.data_transforms_depth = data_transforms_depth
        self.phase = phase
        self.color_loader = color_loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        try:
            img_name = self.img_name[item]
            label = self.img_label[item]
            rgb_img, hsv_img = self.color_loader(img_name)

            if self.data_transforms_color is not None:
                try:
                    rgb_img = self.data_transforms_color(rgb_img)
                    hsv_img = self.data_transforms_color(hsv_img)
                    img = torch.cat([rgb_img, hsv_img], 0)
                except:
                    print("Cannot transform image: {}".format(img_name))
            return img, label
            # return rgb_img, depth_img, label
        except:
            print(item)