import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from myutils.dice_score import multiclass_dice_coeff, dice_coeff
from myutils.mae import calculate_mae
from myutils.metrics import Evaluator
from datasets.utils import decode_segmap
import os
import cv2

# 该函数用于在验证集上评估模型的性能，使用的是Dice系数作为评价指标
@torch.inference_mode()  # 该函数用于在验证集上评估模型的性能，使用的是Dice系数作为评价指标
def evaluate(net, dataloader, n_classes, device):
    net.eval()  # 将模型设置为评估模式，这会影响像dropout或batch normalization这样的层的行为
    num_val_batches = len(dataloader)   # 获取验证集中的批次数量
    dice_score = 0  # 初始化Dice分数
    mae = 0
    evaluator = Evaluator(n_classes)
    evaluator.reset()
    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    
    # 遍历验证集
    
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['label']
        # 将图像和标签移动到正确的设备上，并转换为适当的类型
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        #  预测掩码（mask）
        mask_pred = net(image)
        
        evaluator.add_batch(mask_true.cpu().numpy(), np.argmax(mask_pred.data.cpu().numpy(), axis=1))
        # 如果是二分类问题
        if n_classes == 2:
            # # 确保真实标签的值在[0, 1]之间
            # mask_pred = mask_pred[:, 1, :, :]
            # assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()   # 使用sigmoid激活函数并阈值化，得到预测的二进制掩码
            # # 计算Dice系数
            # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            
            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, 1]'
            # 将标签转换为one-hot格式
            mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            
            # 计算多分类的Dice系数，忽略背景类（即跳过第一个类）
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            mae += calculate_mae(mask_pred, mask_true)
            
        else:
             # 如果是多分类问题
            # 确保真实标签的值在正确的类别范围内
            # print(mask_pred.shape, mask_true.shape)
            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes]'
            # 将标签转换为one-hot格式
            mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            # print(mask_pred.shape, mask_true.shape)
            # print(mask_pred[:, 1:].shape, mask_true[:, 1:].shape)
            # print(mask_pred[:, 1:].flatten(0, 1).shape, mask_true[:, 1:].flatten(0, 1).shape)
            # 计算多分类的Dice系数，忽略背景类（即跳过第一个类）
            # dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            # 计算多分类的Dice系数，不忽略背景类
            dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            mae += calculate_mae(mask_pred, mask_true)
        
    net.train()  # 重新将模型设置为训练模式
    return dice_score / max(num_val_batches, 1), evaluator, mae / max(num_val_batches, 1)  # 返回平均Dice分数，防止除以零

def visualize(args, net, dataloader, n_classes, device):
    net.eval()  # 将模型设置为评估模式，这会影响像dropout或batch normalization这样的层的行为
    num_val_batches = len(dataloader)   # 获取验证集中的批次数量
    
    save_dir = args.test_save_dir
    image_dir = os.path.join(save_dir, 'image')
    label_dir = os.path.join(save_dir, 'label')
    pred_dir = os.path.join(save_dir, 'pred')
    mix_dir = os.path.join(save_dir, 'mix')
    comb_dir = os.path.join(save_dir, 'comb')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(comb_dir, exist_ok=True)
    
    # 遍历验证集    
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, name = batch['image'], batch['label'], batch['name'][0]
        # 将图像和标签移动到正确的设备上，并转换为适当的类型
        raw_image = image.numpy()[0]
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        #  预测掩码（mask）
        mask_pred = net(image)        
        
        if n_classes == 2:
            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, 1]'            
        else:
            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes]'

        mask_pred = mask_pred.argmax(dim=1)                    
        
        # draw images
        
        mask_true = mask_true.cpu().numpy()[0]
        mask_pred = mask_pred.cpu().numpy()[0]
        # mask_true = mask_true
        # print(np.unique(mask_true))
        mask_true = (decode_segmap(mask_true, dataset=args.dataset)*255).astype(np.uint8)
        
        # mask_pred = mask_pred
        # print(np.unique(mask_pred))
        mask_pred = (decode_segmap(mask_pred, dataset=args.dataset)* 255).astype(np.uint8)
        # print(np.sum(mask_true), np.sum(mask_pred))
        raw_image = np.transpose(raw_image, axes=[1, 2, 0])        
        # cv2 only
        raw_image = raw_image[:, :, ::-1]
        if args.dataset == 'Huaxi':
            raw_image *= 255.0
        raw_image = raw_image.astype(np.uint8)
        # print(np.sum(mask_true))
        # print(mask_true.shape)
        alpha = 0.8  #融合比例系数
        mix = (raw_image * alpha + mask_pred * (1 - alpha)).astype(np.uint8)
        # print(np.unique(raw_image), np.unique(mask_true))
        # print('raw_image.dtype', raw_image.dtype)
        # print('mask_true.dtype', mask_true.dtype)
        cv2.imwrite(os.path.join(image_dir, name), raw_image)
        cv2.imwrite(os.path.join(label_dir, name), mask_true)
        cv2.imwrite(os.path.join(pred_dir, name), mask_pred)
        cv2.imwrite(os.path.join(mix_dir, name), mix)
        combined_image = cv2.hconcat([raw_image, mask_true, mask_pred, mix])
        cv2.imwrite(os.path.join(comb_dir, name), combined_image)
        
    return 