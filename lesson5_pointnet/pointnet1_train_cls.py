"""
pointnet 点云分类模型训练
"""
import torch
from tqdm import tqdm
from data_utils.ModelNetDataSet import ModelNetDataSet
import numpy as np

from pointnet1_cls import PointNetCls, PointNetLoss


def main():
    # 超参数设置
    USE_CUDA = True
    NUM_CLASS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DECAY_RATE = 0.0001
    EPOCH = 10

    # 数据加载
    data_path = "/home/mitom/3DPointCloud/data/modelnet40_normal_resampled"
    train_dataset = ModelNetDataSet(root=data_path, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

    classifier = PointNetCls(k=NUM_CLASS)
    critertion = PointNetLoss()

    if USE_CUDA:
        classifier = classifier.cuda()
        loss = critertion.cuda()

    # ADAM
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=DECAY_RATE)
    # SGD
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # 训练
    for epoch in range(EPOCH):
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            optimizer.zero_grad()

            points = points[:,:,0:3]
            points = points.transpose(2,1)
            if USE_CUDA:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = critertion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        print("Train Instance Accuracy: %f" % train_instance_acc)


if __name__ == "__main__":
    main()