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
EPOCH = 10


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
    logger.info("Parameter ...")
    print("Parameter ...")


    # 数据加载
    logger.info("Load dataset ...")
    print("Load dataset ...")

    data_path = "/home/mitom/3DPointCloud/data/modelnet40_normal_resampled"
    train_dataset = ModelNetDataSet(root=data_path, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True) # drop_last=True:最后一个batch数据不完整就删除

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
            print(pred.shape)
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
            pass


def test(model, loader, num_class=NUM_CLASS):
    """

    :param model:
    :param loader:
    :param num_class:
    :return:
    """

    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # (40, 3)
    classifier = model.eval()

    for idx, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if USE_CUDA:
            points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1) # (B,N,3) -> (B,3,N)
        pred, _ = classifier(points) # (B,N)
        pred_choise = pred.data.max(1)[1]

        for category in np.unique(target.cpu()):
            pass





if __name__ == "__main__":
    # main()
    # test(1,1)
    class_acc = np.zeros((5, 3))
    pred_choise = torch.tensor([0,1,2,3,4,0,1,1,2,3])
    target = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    target_u = np.unique(target.cpu())
    print(target_u)
    for cat in target_u:
        classacc = pred_choise[target == cat].eq(target[target==cat].long().data).cpu().sum()
        class_acc[cat, 0] += classacc.item() / float(target[target == cat].size()[0])
        class_acc[cat, 1] += 1
        print(classacc)
    print(class_acc)
    correct = pred_choise.eq(target.long().data).cpu().sum()
    print(correct)
