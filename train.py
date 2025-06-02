import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime


from trainer_huaxi import trainer_huaxi
from config import get_config
#from fvcore.nn import FlopCountAnalysis,flop_count,flop_count_str

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/workspace/OesopStomach/huaxi_seg/First_batch_of_data/', help='root dir for data')
# parser.add_argument('--volume_path', type=str,
#                     default='/mnt/NVME/Segmentation/data/MyThingData', help='root dir for validation volume data') 
parser.add_argument('--dataset', type=str,
                    default='Huaxi', help='experiment_name')
# parser.add_argument('--test_path', type=str,
#                     default='/mnt/NVME/Segmentation/code/Other_result/data/CVC-ClinicDB', help='root dir for data')
# parser.add_argument('--list_dir', type=str,
#                     default='/mnt/NVME/Segmentation/data/MyThingData', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
# parser.add_argument('--output_dir', type=str,default='./model_out', help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.00005,
                    help='segmentation network learning rate')#0.01 or 0.5
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
parser.add_argument('--model', type=str, default='GCtx_UNet')


args = parser.parse_args()

# assert args.dataset == "Huaxi"
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")
# elif args.dataset == "Huaxi":
#     args.root_path = os.path.join(args.root_path, "train")
#     args.volume_path = os.path.join(args.volume_path, "val")
# elif args.dataset == "CVC":
#     args.root_path = os.path.join(args.root_path, "train")
#     args.volume_path = os.path.join(args.volume_path, "val")
config = get_config(args)


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
            'num_classes': args.num_classes,
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

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']

    args.output_dir = args.model + '/' + str(datetime.now())[:-7]
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

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
    
    
    
    net = ViT_seg(config,img_size=args.img_size, num_classes=args.num_classes).cuda()
    # net = UNet_Attention_Transformer_Multiscale(n_channels = 3, n_classes = 3, bilinear=True)
    #inputs = (torch.randn((10,3,224,224)).cuda(),)
    #with open('model_complexity.txt', 'w') as output:
         #output.write(flop_count_str(FlopCountAnalysis(net, inputs)))
    net.load_from(config)
    
    # trainer = {'Huaxi': trainer_huaxi, }
    # trainer = {'CVC': trainer_huaxi,}
    trainer_huaxi(args, net, args.output_dir)
