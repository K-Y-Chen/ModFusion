import os
import random
import glob
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from PIL import Image
# import tifffile as jpgf
import torch.nn.functional as F
# import tifffile as tiff
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from datasets.utils import decode_segmap
# from utils import decode_segmap


# 随机旋转和翻转图像和标签
def random_rot_flip(image, label):
    # 随机旋转0, 90, 180, 270度
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k)
    # 随机翻转图像
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis + 1).copy()  # 因为我们要在 (H, W) 平面翻转
    label = np.flip(label, axis=axis).copy()
    return image, label

# 随机旋转图像和标签，旋转角度在[-20, 20]度之间
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=3)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 数据增强类
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 输出图像的大小

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机选择一种增强方式：旋转翻转或旋转
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 如果图像大小与目标大小不同，则进行缩放
        c, h, w = image.shape
        # print(h, w)
        # print(image.shape)
        # print(self.output_size)
        if (h, w) != self.output_size:
            zoom_factors = (1, self.output_size[0] / h, self.output_size[1] / w)
            # print(zoom_factors)
            image = zoom(image, zoom_factors, order=3)  # 不缩放通道维度
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        # 将numpy数组转换为PyTorch张量，并增加一个维度（通道数）
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        # print(label.shape)
        # 将标签转换为长整型
        sample = {'image': image, 'label': label.long()}
        return sample


# 数据增强类
class EvalRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 输出图像的大小

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 如果图像大小与目标大小不同，则进行缩放
        c, h, w = image.shape
        if (h, w) != self.output_size:
            zoom_factors = (1, self.output_size[0] / h, self.output_size[1] / w)
            image = zoom(image, zoom_factors, order=3)  # 不缩放通道维度
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        # 将numpy数组转换为PyTorch张量，并增加一个维度（通道数）
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        # 将标签转换为长整型
        sample = {'image': image, 'label': label.long()}
        return sample


class ETIS(Dataset):
    def __init__(self, base_dir, num_classes=2, split='train', transform=None):
        # self.transform = transform  # 数据增强变换
        # self.split = split  # 数据集划分（训练或测试）
        # # 读取数据列表
        # self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        # self.data_dir = base_dir  # 数据目录
        self.transform = transform  # 数据增强变换
        output_size=[224, 224]
        self.split = split  # 数据集划分（训练或测试）
        self.num_classes = num_classes
        self.dataset_path = base_dir        #数据路径
        self.dict = {}
        # self.color2id = {50:1, 100:2, 150:3, 200:4, 250:5}
        # if self.num_classes == 4:
        #     self.color2id = {50:1, 100:2, 150:3, 200:0, 250:0}
        # self.m2id = {'dianran':0, 'NBI':1,  'baiguang':2}
        # self.modality = ['dianran', 'NBI',  'baiguang']
        self.dataset2id = {'CVC_Clinic':0, 'ETIS':1, 'kvasir':2}
        self.dataset = 'ETIS'
        self.THRED = 100
        self.num_dataset = 3
        self.image_list = []
        self.label_list = []
        assert split == 'train' or split == 'test' or split == 'validation'
        
        images = glob.glob(os.path.join(base_dir, split, 'images', '*'))
        
        for idx in range(len(images)):
            # print(images[idx], gts[idx])
            basename = os.path.basename(images[idx])
            self.image_list.append(os.path.join(base_dir, split, 'images', basename))
            self.label_list.append(os.path.join(base_dir, split, 'masks', basename))
                
        self.train_batches = len(self.image_list)# train_batches
        self.init_transform = transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)
        
    def __len__(self):
        return len(self.image_list)  # 返回数据集大小

    def __getitem__(self, idx):
        # if self.split == "train":
        #     # 获取训练数据
        #     slice_name = self.sample_list[idx].strip('\n')

        #     image_path = os.path.join(self.data_dir, 'images', slice_name + '.jpg')
        #     label_path = os.path.join(self.data_dir, 'masks', slice_name + '.png')

        # else:
        #     # 获取测试数据
        #     vol_name = self.sample_list[idx].strip('\n')

        #     image_path = os.path.join(self.data_dir, 'images', vol_name + '.jpg')
        #     label_path = os.path.join(self.data_dir, 'masks', vol_name + '.png')
        image_path = self.image_list[idx]
        image_name = os.path.basename(image_path)        
        label_path = self.label_list[idx]
        # 读取彩色图像和灰度标签
        # print(image_path)
        
        image = np.array(self.init_transform(Image.open(image_path)))
        label = np.array(self.init_transform(Image.open(label_path).convert('L')))
        # image = np.array(Image.open(image_path))
        # label = np.array(Image.open(label_path).convert('L'))
        # print(np.unique(label))
        # print(image.shape)
        # for k in self.color2id.keys():
        #     # print(k)
        #     label[label==k] = self.color2id[k]
        # image = jpgf.imread(image_path)
        # label = jpgf.imread(label_path)

        
        # label = np.expand_dims(label, axis=0)  # 增加一个维度 (1, H, W)

        # 读取彩色图像和单通道标签
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # label[label == 255] = 1
        # label[label != 1] = 0  # 将所有其他值映射为 0

         # 确保标签范围在 [0, n_classes-1]
        # n_classes = 2  # 根据你的情况设置


        # 转换图像为 (C, H, W) 格式
        image = image.transpose(2, 0, 1)
        label[label < self.THRED] = 0
        label[label >= self.THRED] = 1
        # print(np.unique(label))
        
        
        sample = {'image': image, 'label': label}

        cls_label = self.dataset2id[self.dataset]
        cls_label = torch.tensor(cls_label).long()
        cls_label = F.one_hot(cls_label, self.num_dataset).float()
        
        if self.transform:
            sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        # sample["path"] = image_path
        sample['name'] = image_name
        sample['cls'] = cls_label
        return sample
    
    # def decode_segmap(self, label_mask, plot=False):
    #     """Decode segmentation class labels into a color image
    #     Args:
    #         label_mask (np.ndarray): an (M,N) array of integer values denoting
    #         the class label at each spatial location.
    #         plot (bool, optional): whether to show the resulting color image
    #         in a figure.
    #     Returns:
    #         (np.ndarray, optional): the resulting decoded color image.
    #     """
        
    #     n_classes = self.num_classes
    #     label_colours = np.array([
    #     [0, 0, 0],          # background
    #     [255, 255, 255],
    #     ])

    #     r = label_mask.copy()
    #     g = label_mask.copy()
    #     b = label_mask.copy()
    #     for ll in range(0, n_classes):
    #         r[label_mask == ll] = label_colours[ll, 0]
    #         g[label_mask == ll] = label_colours[ll, 1]
    #         b[label_mask == ll] = label_colours[ll, 2]
    #     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    #     rgb[:, :, 0] = r / 255.0
    #     rgb[:, :, 1] = g / 255.0
    #     rgb[:, :, 2] = b / 255.0
    #     if plot:
    #         plt.imshow(rgb)
    #         plt.show()
    #     else:
    #         return rgb

# class cvc_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None, num_slices=2):
#         self.transform = transform  # 数据增强变换
#         self.split = split  # 数据集划分（训练或测试）
#         # 读取数据列表
#         self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
#         self.data_dir = base_dir  # 数据目录

#     def __len__(self):
#         return len(self.sample_list)  # 返回数据集大小

#     def __getitem__(self, idx):
#         if self.split == "train":
#             # 获取训练数据
#             slice_name = self.sample_list[idx].strip('\n')

#             image_path = os.path.join(self.data_dir, self.split, 'images', slice_name + '.tif')
#             label_path = os.path.join(self.data_dir, self.split, 'masks', slice_name + '.tif')

#         else:
#             # 获取测试数据
#             vol_name = self.sample_list[idx].strip('\n')

#             image_path = os.path.join(self.data_dir, self.split, 'images', vol_name + '.tif')
#             label_path = os.path.join(self.data_dir, self.split, 'masks', vol_name + '.tif')
            
#         # 读取彩色图像和灰度标签
#         # # print(image_path)
#         # image = np.array(Image.open(image_path))
#         # label = np.array(Image.open(label_path).convert('L'))
#         image = tiff.imread(image_path)
#         label = tiff.imread(label_path)

        
#         # label = np.expand_dims(label, axis=0)  # 增加一个维度 (1, H, W)

#         # 读取彩色图像和单通道标签
#         # image = cv2.imread(image_path)
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
#         # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#         label[label == 255] = 1
#         label[label != 1] = 0  # 将所有其他值映射为 0

#         # 确保标签范围在 [0, n_classes-1]
#         # n_classes = 2  # 根据你的情况设置


#         # 转换图像为 (C, H, W) 格式
#         image = image.transpose(2, 0, 1)
#         sample = {'image': image, 'label': label}

#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         # sample["path"] = image_path
#         return sample

if __name__ == '__main__':
    import argparse

    import matplotlib.pyplot as plt
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    # from torchvision import transforms
    # import sys
    # sys.path.append("/workspace/OesopStomach/huaxi_seg/code/pytorch-deeplab-xception")

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'ETIS'
    args.base_dir = '/workspace/huaxicolor/Datasets/ETIS-LaribPolypDB'
    
    train = ETIS(args.base_dir, num_classes=2, split='validation', transform=transforms.Compose([RandomGenerator(output_size=[224, 224])]))
    # train = ETIS(args.base_dir, num_classes=2, split='validation', transform=None)
    print(len(train))
    print('NUM_CLASSES = ', train.num_classes)
    # print(train[0])
    dataloader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            # print(np.unique(gt))
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, args.dataset)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # img_tmp = img_tmp[:, :, ::-1]
            # img_tmp *= (0.229, 0.224, 0.225)
            # img_tmp += (0.485, 0.456, 0.406)
            # img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            # print(np.sum(segmap))
            print(np.unique(segmap))
            print(segmap.shape)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        # if ii == 1:
        #     break
    # plt.show()
        plt.savefig(f'./display.{args.dataset}_{ii}.png')