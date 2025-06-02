import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets.dataset_huaxi import huaxi_dataset
from myutils.evaluate import evaluate, visualize
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/workspace/OesopStomach/huaxi_seg/First_batch_of_data/', help='root dir for data')
# parser.add_argument('--volume_path', type=str,
#                     default='/mnt/NVME/Segmentation/data/seg_cancer', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Huaxi', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='/mnt/NVME/Segmentation/data/seg_cancer', help='list dir')
parser.add_argument('--ckpt_dir', type=str, default=None, required=True, help='')
parser.add_argument('--checkpoint', '-ckpt', type=str, required=True, help='loading checkpoint')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--visual', default=False, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions_huaxi/', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=False, default='configs/GCViT_xxtiny_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument('--checkpoint', '-ckpt', type=str)

parser.add_argument('--model', type=str, default='GCtx_UNet')

args = parser.parse_args()
# assert args.dataset == "Huaxi"
# if args.dataset == "Synapse":
#     args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# elif args.dataset == "Huaxi":
#     args.volume_path = os.path.join(args.volume_path, "val")
config = get_config(args)


# def inference(args, model, test_save_path=None):
#     from datasets.dataset_huaxi import huaxi_dataset, RandomGenerator
#     db_test = args.Dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir)

#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     counter = 0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         # print(sampled_batch["image"].shape)
#         # exit()
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, img_path, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['path'], sampled_batch['case_name'][0]
#         final_test_single_volume(img_path, image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#     #     if (1,0) in metric_i:
#     #         counter = counter + 1
#     #         print(" ------------------counter: ", counter, "metric_i: ", metric_i)
#     #     metric_list += np.array(metric_i)
#     #     logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     # metric_list = metric_list / len(db_test)
#     # for i in range(1, args.num_classes):
#     #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     # performance = np.mean(metric_list, axis=0)[0]
#     # mean_hd95 = np.mean(metric_list, axis=0)[1]
#     # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"


# def train_inference(args, model, dataloader, test_save_path=None):
#     # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
#     # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(dataloader)))
#     model.eval()
#     # metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
#         image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#         image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#         outputs = model(image_batch)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i = final_test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Huaxi': {
            'root_path': '/workspace/OesopStomach/huaxi_seg/First_batch_of_data/',            
            'num_classes': 4,
        },
        
        'Kvasir': {
            'root_path': '/workspace/huaxicolor/Datasets/KvasirSEG',            
            'num_classes': 2,
        },
        
        'CVC_Clinic': {
            'root_path': '/workspace/huaxicolor/Datasets/CVC-ClinicDB',            
            'num_classes': 2,
        },
        
        'ETIS': {
            'root_path': '/workspace/huaxicolor/Datasets/ETIS-LaribPolypDB',            
            'num_classes': 2,
        },
        # 'CVC': {
        #     'root_path': args.root_path,
        #     'list_dir': '/mnt/NVME/Segmentation/code/Other_result/data/CVC-ClinicDB',
        #     'num_classes': 2,
        # },
    }
    
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    # args.dataset = dataset_config[dataset_name]['Dataset']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = True
    
    
    if args.model.__contains__('GCtx'):
        if args.model == 'GCtx_UNet':
            from networks.GCtx_UNet import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv1':
            from networks.GCtx_UNet_FDv1 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv2':
            from networks.GCtx_UNet_FDv2 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv3':    
            from networks.GCtx_UNet_FDv3 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv4':    
            from networks.GCtx_UNet_FDv4 import GCViT_Unet as ViT_seg 
        elif args.model == 'GCtx_UNet_FDv5':    
            from networks.GCtx_UNet_FDv5 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v6':    
            from networks.GCtx_UNet_v6 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv7':    
            from networks.GCtx_UNet_FDv7 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv8':    
            from networks.GCtx_UNet_FDv8 import GCViT_Unet as ViT_seg    
        elif args.model == 'GCtx_UNet_FDv9':    
            from networks.GCtx_UNet_FDv9 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv10':    
            from networks.GCtx_UNet_FDv10 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv11':    
            from networks.GCtx_UNet_FDv11 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_FDv12':    
            from networks.GCtx_UNet_FDv12 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v13':    
            from networks.GCtx_UNet_v13 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v14':    
            from networks.GCtx_UNet_v14 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v15':    
            from networks.GCtx_UNet_v15 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v16':    
            from networks.GCtx_UNet_v16 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v17':    
            from networks.GCtx_UNet_v17 import GCViT_Unet as ViT_seg
        elif args.model == 'GCtx_UNet_v18':    
            from networks.GCtx_UNet_v18 import GCViT_Unet as ViT_seg 
        elif args.model == 'GCtx_UNet_v19':    
            from networks.GCtx_UNet_v19 import GCViT_Unet as ViT_seg 
        elif args.model == 'GCtx_UNet_v20':    
            from networks.GCtx_UNet_v20 import GCViT_Unet as ViT_seg 
        else:
            raise NotImplementedError(f'{args.model} Not Implemented.')
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    else:
        if args.model == 'PraNet':
            from others.PraNet.PraNet_Res2Net import PraNet as model
        elif args.model == 'PSPNet':
            from others.PSPNet.pspnet import PSPNet as model
        elif args.model == 'EMCAD':
            from others.EMCAD.networks import EMCADNet as model
        elif args.model == 'FCBFormer':
            from others.FCBFormer.models import FCBFormer as model
        else:
            raise NotImplementedError(f'{args.model} Not Implemented.')    
        net = model(num_classes=args.num_classes).cuda()
    
    batch_size = args.batch_size * args.n_gpu
    if args.visual:
        batch_size = 1
        
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    if args.dataset == 'Huaxi':
        from datasets.dataset_huaxi import huaxi_dataset, RandomGenerator, EvalRandomGenerator
        db_train = huaxi_dataset(base_dir=args.root_path, num_classes=args.num_classes, split='train',
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        print("The length of train set is: {}".format(len(db_train)))
        
        db_val = huaxi_dataset(base_dir=args.root_path, num_classes=args.num_classes, split='val',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
        
        db_test = huaxi_dataset(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)

    elif args.dataset == 'Kvasir':
        from datasets.kvasir import KvasirSEG, RandomGenerator, EvalRandomGenerator
        db_train = KvasirSEG(base_dir=args.root_path, num_classes=args.num_classes, split='train',
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        print("The length of train set is: {}".format(len(db_train)))
        
        db_val = KvasirSEG(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
        
        db_test = KvasirSEG(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)

    elif args.dataset == 'CVC_Clinic':
        from datasets.CVC_Clinic import CVC_Clinic, RandomGenerator, EvalRandomGenerator
        db_train = CVC_Clinic(base_dir=args.root_path, num_classes=args.num_classes, split='train',
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        print("The length of train set is: {}".format(len(db_train)))
        
        db_val = CVC_Clinic(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
        
        db_test = CVC_Clinic(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)
        
    elif args.dataset == 'ETIS':
        from datasets.ETIS import ETIS, RandomGenerator, EvalRandomGenerator
        db_train = ETIS(base_dir=args.root_path, num_classes=args.num_classes, split='train',
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        print("The length of train set is: {}".format(len(db_train)))
        
        db_val = ETIS(base_dir=args.root_path, num_classes=args.num_classes, split='validation',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)
        
        db_test = ETIS(base_dir=args.root_path, num_classes=args.num_classes, split='test',
                                transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
        testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    else:
        raise NotImplementedError()    
    # snapshot = os.path.join(args.output_dir, 'epoch_299.pth')# epoch_149.pth
    # snapshot = os.path.join(args.output_dir, 'best.pth')# epoch_149.pth
    snapshot = os.path.join(args.ckpt_dir, args.checkpoint)
    
    if not os.path.exists(snapshot): 
        # snapshot = snapshot.replace('epoch_299', 'epoch_'+str(args.max_epochs-1))
        print(f"model {snapshot} not exits")
        exit()
    msg = net.load_state_dict(torch.load(snapshot), strict=False)
    print(f"self trained {args.model}: {msg}")
    snapshot_dir = args.ckpt_dir
    snapshot_name = args.checkpoint

    log_folder = os.path.join(args.ckpt_dir, './test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # logging.info(snapshot_name)

    if args.visual:
        # print(f"Drawing images from: {snapshot}")
        logging.info(f"Drawing images from: {snapshot}")
        args.test_save_dir = os.path.join(snapshot_dir, "predictions")
        os.makedirs(args.test_save_dir, exist_ok=True)
        # print(f"Save images to: {args.test_save_dir}")
        logging.info(f"Save images to: {args.test_save_dir}")
        visualize(args, net, valloader, n_classes=valloader.dataset.num_classes, device='cuda')
        
    else:           
        val_score, evaluator, mae = evaluate(net, valloader, n_classes=valloader.dataset.num_classes, device='cuda')        
        
        is_best_dice = False
        is_best_iou = False
        # Acc = evaluator.Pixel_Accuracy()
        # Acc_class = evaluator.Pixel_Accuracy_Class()
        # mIoU = evaluator.Mean_Intersection_over_Union()
        # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        
        logging.info(f"Testing from: {snapshot}")
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        P = evaluator.Precision()
        R = evaluator.Recall()
        F1 = evaluator.F_score()
        logging.info("Validate Dice:{}".format(val_score))
        logging.info("MAE:{}".format(mae))
        logging.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        logging.info("Precision:{}, Recall:{}, F1-Score:{}".format(P, R, F1))
    logging.info("------------------------------------------------------------------------------------------------------")
    
    
    
    
    
    
    
    
    
    


