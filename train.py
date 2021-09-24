import numpy as np
import paddle
#import torch.backend.cudnn as cudnn
import paddle.optimizer as optim
from paddle.io import DataLoader

from model.nets.yolo import YoloBody
from model.nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #是否使用GPU
    Cuda = False

    #分类文件和模型文件
    classes_path = 'datasets/voc_classes.txt'
    model_path = 'weights/yolox_s.pdparams'
    #使用的YoloX版本：s、m、l、x
    phi = 's'

    #shape大小
    input_shape = [640, 640]

    #马赛克数据增强
    mosaic = False
    #余弦退火学习率
    Cosine_scheduler = False

    #冻结阶段训练参数，模型主干冻结，特征提取网络不改变，网络微调
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    Freeze_lr = 1e-3

    #解冻阶段训练参数，特征提取网络改变，网络参数会发生变化
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4

    #是否进行冻结训练，默认先冻结训练后解冻训练
    Freeze_Train = True

    #多线程读取数据
    num_workers = 2

    #图片路径和标签
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    #获取classes和anchor
    class_names, num_classes = get_classes(classes_path)

    #创建yolo模型
    model = YoloBody(num_classes, phi)
    weights_init(model)

    #读取yolo网络
    print('Load weights {}.'.format(model_path))
    device = paddle.device.get_device()
    model_dict = model.state_dict()
    pretrained_dict = paddle.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.set_state_dict(model_dict)

    model.train()
    # model_train = model.train()
    # if Cuda:
    #     model_train = paddle.DataParallel(model)
    #     model.train()

    yolo_loss = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/")

    #读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(learning_rate=lr, weight_decay=5e-4, parameters=model.parameters())
        if Cosine_scheduler:
            lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=lr, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr.StepDecay(learning_rate=lr, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train = True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                    train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model, yolo_loss, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
        lr_scheduler.step()

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        optimizer = optim.Adam(learning_rate=lr, weight_decay=5e-4, parameters=model.parameters())
        if Cosine_scheduler:
            lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=lr, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr.StepDecay(learning_rate=lr, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=mosaic,
                                    train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, end_epoch - start_epoch, mosaic=False,
                                  train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model, yolo_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()