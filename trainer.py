import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils import DiceLoss, dice_coefficient_multiclass, calculate_metrics
from utils import DiceLoss
# from myutils.dice_score import dice_loss as DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.optim import lr_scheduler
# from Utils.dice_score import dice_loss
from myutils.evaluate import evaluate
# from .dice_score import dice_loss-+

def trainer_huaxi(args, model, snapshot_path):
    # print(snapshot_path)
    snapshot_path = os.path.join('runs', args.dataset, snapshot_path)
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    # test_save_path = os.path.join(snapshot_path, 'test')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    # print(args.num_classes)
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    
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
    # if args.dataset == 'Huaxi':
    #     db_train = huaxi_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                             transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    #     print("The length of train set is: {}".format(len(db_train)))
    #     db_test = huaxi_dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir,
    #                             transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
    #     testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # elif args.dataset == 'CVC':
    #     db_train = cvc_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                             transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    #     print("The length of train set is: {}".format(len(db_train)))
    #     db_test = cvc_dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir,
    #                             transform=transforms.Compose([EvalRandomGenerator(output_size=[args.img_size, args.img_size])]))
    #     testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=8)

    

    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(trainloader), T_mult=2)
    writer = SummaryWriter(snapshot_path + '/log')
    best_dice, best_iou, best_p, best_r, best_f1 , best_epoch = 0., 0., 0, 0, 0, 0
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(1, max_epoch + 1), desc="Epochs", ncols=70)
    # best_epoch_dice = 0
    # best_epoch_iou = 0
    
    for epoch_num in iterator:
        model.train()
        train_progress = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training Epoch {epoch_num}", ncols=70, leave=False)

        # for i_batch, sampled_batch in enumerate(trainloader):
        for i_batch, sampled_batch in train_progress:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # print(np.unique(label_batch.cpu().numpy()))
            # exit()
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_dice = dice_loss(outputs, label_batch)
            
            loss = 0.3 * loss_ce + 0.7 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # if we used this no need to lr_r Or upadate the lr_base
            iter_num += 1
            if iter_num % 500 == 0:
                # print('iteration %d : loss : %f, loss_dice:%f, loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))
                train_progress.set_postfix(loss=loss.item(), loss_dice=loss_dice.item(), loss_ce=loss_ce.item())
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


        '''
        model.eval()
        dice_list = []
        # avg_precision, avg_recall, avg_f1 = [], [], []
        # if (epoch_num + 1) % 1 == 0:
        test_progress = tqdm(testloader, desc="Testing", ncols=70, leave=False)
        # for sampled_batch in tqdm(testloader):
        for sampled_batch in test_progress:
            image, label = sampled_batch["image"], sampled_batch["label"]
            input = image.float().cuda()
            with torch.no_grad():
                outputs = model(input).cpu().detach()
            
            dice_list.append(dice_coefficient_multiclass(outputs, label))

            # p, r, f1 = calculate_metrics(outputs, label, num_classes)
            # avg_precision.append(p)
            # avg_recall.append(r)
            # avg_f1.append(f1)
        # mean_dice, mean_p, mean_r, mean_f1 = np.mean(dice_list), np.mean(avg_precision), np.mean(avg_recall), np.mean(avg_f1)
        mean_dice = np.mean(dice_list)
        if mean_dice > best_performance:
            best_performance = mean_dice
            best_epoch = epoch_num
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info(f"Epoch num: {epoch_num}, Dice: {mean_dice}")
        logging.info(f'Best Performance: -- best dice: {best_performance} -- best epoch:{best_epoch}')
        # print(f"Epoch num: {epoch_num+1}, Dice: {np.mean(dice_list)}, Precision: {np.mean(avg_precision)}, Recall: {np.mean(avg_recall)}, F1 socre: {np.mean(avg_f1)}")
        
        '''
        # print(num_classes)
        val_score, evaluator, val_mae = evaluate(model, valloader, n_classes=num_classes, device='cuda')        
        scheduler.step(val_score)
        # print(val_score)
        is_best_dice = False
        is_best_iou = False
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        
        if val_score > best_dice:
            best_dice = val_score
            best_epoch_dice = epoch_num
            is_best_dice = True
        
        if mIoU > best_iou:
            best_iou = mIoU
            best_epoch_iou = epoch_num
            is_best_iou = True
            
        # logging.info('Validation Dice score: {}'.format(val_score))
        # logging.info('epoch {} : loss (batch): {:.4f}, Validation Dice score: {}'.format(epoch_num, loss.item(), val_score))
        logging.info(f"Epoch num: {epoch_num}, Validate Dice: {val_score}")
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        P = evaluator.Precision()
        R = evaluator.Recall()
        F1 = evaluator.F_score()
        logging.info("MAE:{}".format(val_mae))
        logging.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        logging.info("Precision:{}, Recall:{}, F1-Score:{}".format(P, R, F1))
        logging.info(f'Best Dice: -- best dice: {best_dice} -- best epoch:{best_epoch_dice}')
        logging.info(f'Best mIoU: -- best miou: {best_iou} -- best epoch:{best_epoch_iou}')
        
        if is_best_dice: # Test dataset
            save_mode_path = os.path.join(snapshot_path, 'dice_best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save best Dice model to {}".format(save_mode_path))
        if is_best_iou:
            save_mode_path = os.path.join(snapshot_path, 'iou_best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save best mIoU model to {}".format(save_mode_path))
        if  is_best_dice or is_best_iou:   
            logging.info("Testing best model")
            test_score, evaluator, test_mae = evaluate(model, testloader, n_classes=num_classes, device = 'cuda')
            logging.info(f" Epoch num: {epoch_num}, Test Dice: {test_score}")
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            P = evaluator.Precision()
            R = evaluator.Recall()
            F1 = evaluator.F_score()
            logging.info("MAE:{}".format(test_mae))
            logging.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            logging.info("Precision:{}, Recall:{}, F1-Score:{}".format(P, R, F1))
        
        
        # if epoch_num >= 30:
        # save latest epoch
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save lastest model to {}".format(save_mode_path))
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    test_score, evaluator, test_mae = evaluate(model, testloader, n_classes=num_classes, device = 'cuda')
    logging.info(f"Final epoch, Test Dice: {test_score}")
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    P = evaluator.Precision()
    R = evaluator.Recall()
    F1 = evaluator.F_score()
    logging.info("MAE:{}".format(test_mae))
    logging.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    logging.info("Precision:{}, Recall:{}, F1-Score:{}".format(P, R, F1))
            
    writer.close()
    return "Training Finished!"
