"""
pointnet 点云分类模型训练
"""

import datetime
import os.path
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from data_utils.ModelNetDataSet import ModelNetDataSet
import numpy as np

from pointnet1_cls import PointNetCls, PointNetClsLoss

""" 超参数设置 """
USE_CUDA = True
NUM_CLASS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DECAY_RATE = 0.0001
EPOCH = 2

params = {"USE_CUDA":USE_CUDA,
          "NUM_CLASS":NUM_CLASS,
          "BATCH_SIZE":BATCH_SIZE,
          "LEARNING_RATE":LEARNING_RATE,
          "DECAY_RATE":DECAY_RATE,
          "EPOCH":EPOCH}


def main():
    # 创建日志目录
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print(timestr)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir()

    # 日志
    logger = logging.getLogger('Model')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "pointnet2_cls_ssg_train"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Parameter %s" % str(params))
    print("Parameter %s" % str(params))


    # 数据加载
    logger.info("Load dataset ...")
    print("Load dataset ...")

    data_path = "/home/mitom/3DPointCloud/data/modelnet40_normal_resampled"
    train_dataset = ModelNetDataSet(root=data_path, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True) # drop_last=True:最后一个batch数据不完整就删除
    test_dataset = ModelNetDataSet(root=data_path, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    classifier = PointNetCls(k=NUM_CLASS)
    critertion = PointNetClsLoss()

    if USE_CUDA:
        classifier = classifier.cuda()
        loss = critertion.cuda()

    # 加载checkpoint
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_start_dict'])
        logger.info("Use pretrain model ...")
        print("Use pretrain model ...")
    except:
        logger.info("No existing model, starting training from epoch=0")
        print("No existing model, starting training from epoch=0")
        start_epoch = 0


    # ADAM
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=DECAY_RATE)
    # SGD
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # 调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # 计数
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    # 训练
    logger.info("Start training ...")
    print("Start training ...")
    for epoch in range(start_epoch, EPOCH):
        logger.info('Epoch %d (%d/%d)' % (global_epoch + 1, epoch + 1, EPOCH))
        print('Epoch %d (%d/%d)' % (global_epoch + 1, epoch + 1, EPOCH))

        # 存储每次精度
        mean_correct = []
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9): # tqdm进度条
            optimizer.zero_grad()

            points = points[:,:,0:3] # 取 x y z
            points = points.transpose(2,1) # (B,N,D) -> (B,D,N)
            if USE_CUDA:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = critertion(pred, target.long(), trans_feat)

            # pred.shape: (B, 40)
            # pred_choice: 返回分数最大的index  (B,)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        logger.info("Train Instance Accuracy: %f" % train_instance_acc)
        print("Train Instance Accuracy: %f" % train_instance_acc)

        # 不计算梯度，对比精度
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), test_dataloader, num_class=NUM_CLASS)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

        if instance_acc >= best_instance_acc:
            logger.info('Save model ...')
            print('Save model ...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            logger.info('Saveing at %s' % savepath)
            print('Saveing at %s' % savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        global_epoch += 1
    logger.info('End of training ...')
    print('End of training ...')


def test(model, loader, num_class=NUM_CLASS):
    """
    验证模型函数
    :param model:
    :param loader:
    :param num_class:
    :return:
    """
    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # (40, 3)
    classifier = model.eval()

    for idx, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = points[:, :, 0:3]  # 取 x y z
        if USE_CUDA:
            points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1) # (B,N,3) -> (B,3,N)
        pred, _ = classifier(points) # (B,N)
        pred_choice = pred.data.max(1)[1]

        for category in np.unique(target.cpu()):
            classacc = pred_choice[target == category].eq(target[target == category].long().data).cpu().sum()
            class_acc[category, 0] += classacc.item() / float(points[target == category].size()[0])
            class_acc[category, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


if __name__ == "__main__":
    main()

