#!/usr/bin/env python3

#导入模块
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse

#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('--EPOCH', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--BATCH_SIZE', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else: raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('--DOWNLOAD_MNIST', type=str2bool, nargs='?', const=True,
                    help='DOWNLOAD_MNIST (default: True)')
parser.add_argument('--TrainOrNot', type=str2bool, nargs='?', const=True,
                    help='TrainOrNot (default: True)')

args = parser.parse_args()

#检测是否有利用的gpu环境
use_gpu = torch.cuda.is_available()
print('use GPU:',use_gpu)

#数据预处理
train_data = torchvision.datasets.MNIST(root='./mnist/',train=True,transform=torchvision.transforms.ToTensor(),download=args.DOWNLOAD_MNIST) #将数据转成tensor结构
train_loader = Data.DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True) #DataLoader是进行批处理的好工具

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)

#测试数据中
if use_gpu:
    test_x = torch.nn.DataParallel(test_x,device_ids=range(torch.cuda.device_count()))
    test_y = torch.nn.DataParallel(test_y, device_ids=range(torch.cuda.device_count()))
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.
    test_y = test_data.test_labels[:2000].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]

#神经网络建模
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

#训练以及保存模型数据
def ModelTrainSave():
    if use_gpu:
        cnn = CNN()
        cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
        cnn = CNN().cuda()
    else:
        cnn =CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.EPOCH):
        for step, (x,y) in enumerate(train_loader):
            if use_gpu:
                b_x = x.cuda()
                b_y = y.cuda()
            else:
                b_x = x
                b_y = y
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad() #将上一步梯度值清零
            loss.backward() #求此刻各参数的梯度值
            optimizer.step() #更新参数

            if step % 50 == 0:
                test_output = cnn(test_x)
                if use_gpu:
                    pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
                    pred_y = torch.max(test_output, 1)[1].cuda().data
                else:
                    pred_y = torch.max(test_output, 1)[1].data
                print(pred_y)
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor)/test_y.size(0)
                print('Epoch',epoch,'| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    torch.save(cnn,'cnn_entire.pkl')
    torch.save(cnn.state_dict(),'net_params.pkl')

#加载模型和训练好的参数
def CNNModerRestore():
    if use_gpu:
        cnnrestore = torch.load('cnn_entire.pkl')
        cnnrestore = torch.nn.DataParallel(cnnrestore, device_ids=range(torch.cuda.device_count()))
        cnnrestore = torch.load('cnn_entire.pkl').cuda()
    else:
        cnnrestore = torch.load('cnn_entire.pkl')
    test_output = cnnrestore(test_x)
    if use_gpu:
        pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
        pred_y = torch.max(test_output, 1)[1].cuda().data
    else:
        pred_y = torch.max(test_output, 1)[1].data
    print(pred_y)
    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print( 'test accuracy: %.2f' % accuracy)

#只加载训练好的参数
def CNNParams():
    if use_gpu:
        cnnparams = CNN()
        cnnparams = torch.nn.DataParallel(cnnparams, device_ids=range(torch.cuda.device_count()))
        cnnparams = CNN().cuda()
    else:
        cnnparams = CNN()
    cnnparams.load_state_dict(torch.load('net_params.pkl'))
    test_output = cnnparams(test_x)
    if use_gpu:
        pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
        pred_y = torch.max(test_output, 1)[1].cuda().data
    else:
        pred_y = torch.max(test_output, 1)[1].data
    print(pred_y)
    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print( 'test accuracy: %.2f' % accuracy)

#运行训练以及保存模型
if (__name__ == '__main__') and args.TrainOrNot:
    ModelTrainSave()

if (__name__ == '__main__') and (not args.TrainOrNot):
    CNNModerRestore()
    CNNParams()
