#!/usr/bin/env python3

#导入模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import argparse
import os
from tensorboardX import SummaryWriter


#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('--EPOCH', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--BATCH_SIZE', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--MODELFOLDER',type= str, default='./model/',
                help="folder to store model")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else: raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser.add_argument('-d','--DOWNLOAD_MNIST', type=str2bool, nargs='?', const=True, required=True,
                    help='DOWNLOAD_MNIST (default: True)')
parser.add_argument('-t','--TrainOrNot', type=str2bool, nargs='?', const=True, required= True,
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
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.
    test_y = test_data.test_labels[:2000].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.test_labels[:2000]

#神经网络建模
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.max_pool2d(output, 2)
        output = F.relu(self.conv2(output))
        output = F.max_pool2d(output, 2)
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16,32,5,1,2),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.out = nn.Linear(32*7*7,10)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0),-1)
#         output = self.out(x)
#         return output

#训练以及保存模型数据
def ModelTrainSave():
    if os.path.exists(
            args.MODELFOLDER + 'net_params_best.pth'):
        print('reload the last best model parameters')
        if use_gpu:
            cnn = LeNet()
            cnn.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_best.pth'))
#            cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
#            cudnn.benchmark = True
            cnn = cnn.cuda()
        else:
            cnn = LeNet()
            cnn.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_best.pth'))
    else:
        if use_gpu:
            cnn = LeNet()
#            cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
#            cudnn.benchmark = True
            cnn = cnn.cuda()
        else:
            cnn = LeNet()
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

            # 可视化模型结构
            with SummaryWriter(log_dir='cnn01') as w:
                #这里也有一些问题，貌似b_x
                w.add_graph(cnn,(b_x,))
                w.add_scalar('Train', loss, global_step=(epoch+1)*100+step )

            if step % 50 == 0:
                test_output = cnn(test_x)
               
                pred_y = torch.max(test_output, 1)[1].data
                print(pred_y)
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor)/test_y.size(0)
                print('Epoch',epoch,'| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
        #检查是否有模型文件夹，没有就自行创建一个
        if not os.path.isdir(args.MODELFOLDER):
            os.makedirs(args.MODELFOLDER)
        #保存一个周期训练好的模型以及参数，并且周期数要记录
        torch.save(cnn,args.MODELFOLDER+'cnn_entire'+str(args.EPOCH+1)+'.pth')
        torch.save(cnn.state_dict(),args.MODELFOLDER+'net_params'+str(args.EPOCH+1)+'.pth')
        #保存最新训练好的模型和参数，为日后加载使用
        torch.save(cnn, args.MODELFOLDER + 'cnn_entire_new.pth')
        torch.save(cnn.state_dict(), args.MODELFOLDER + 'net_params_new.pth')
        #比较最新训练好的参数和上一次训练最好的参数，然后保存最优的
        if epoch ==0:
            torch.save(cnn, args.MODELFOLDER + 'cnn_entire_best.pth')
            torch.save(cnn.state_dict(), args.MODELFOLDER + 'net_params_best.pth')

        if os.path.isfile(args.MODELFOLDER + 'cnn_entire_new.pth') and os.path.isfile(args.MODELFOLDER + 'cnn_entire_best.pth'):
            if use_gpu:
                cnnNew= torch.load(args.MODELFOLDER + 'cnn_entire_new.pth')
#                cnnNew = torch.nn.DataParallel(cnnNew, device_ids=range(torch.cuda.device_count()))
                cnnNew = cnnNew.cuda()
            else:
                cnnNew = torch.load(args.MODELFOLDER + 'cnn_entire_new.pth')
            if use_gpu:
                cnnorg= torch.load(args.MODELFOLDER + 'cnn_entire_best.pth')
#                cnnorg = torch.nn.DataParallel(cnnorg, device_ids=range(torch.cuda.device_count()))
                cnnorg = cnnorg.cuda()
            else:
                cnnorg = torch.load(args.MODELFOLDER + 'cnn_entire_best.pth')

            test_output_New = cnnNew(test_x)
            test_output_org = cnnorg(test_x)
            if use_gpu:
             #   pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
                predNew_y = torch.max(test_output_New, 1)[1].cuda().data
                predorg_y = torch.max(test_output_org, 1)[1].cuda().data
            else:
                predNew_y = torch.max(test_output_New, 1)[1].data
                predorg_y = torch.max(test_output_org, 1)[1].data
            accuracyNew = torch.sum(predNew_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            accuracyorg = torch.sum(predorg_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch',epoch, '| new accuracy: %.5f' % accuracyNew, '| org accuracy: %.5f' % accuracyorg)
            if accuracyNew > accuracyorg:
                #这样写有问题，就是会把new的训练参数删掉，需要进一步修改
                os.renames(args.MODELFOLDER + 'cnn_entire_new.pth',args.MODELFOLDER + 'cnn_entire_best.pth')
                print('best model save update in epoch', epoch)
            else:
                print('best model is still the original')

        if os.path.isfile(args.MODELFOLDER + 'net_params_new.pth') and os.path.isfile(args.MODELFOLDER + 'net_params_best.pth'):
           if use_gpu:
               cnnpnew = LeNet()
               cnnpnew.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_new.pth'))
#                cnnpnew = torch.nn.DataParallel(cnnpnew, device_ids=range(torch.cuda.device_count()))
               cnnpnew = cnnpnew.cuda()
           else:
               cnnpnew  = LeNet()
               cnnpnew.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_new.pth'))
           if use_gpu:
               cnnpo = LeNet()
               cnnpo.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_best.pth'))
#                cnnpo= torch.nn.DataParallel(cnnpo, device_ids=range(torch.cuda.device_count()))
               cnnpo = cnnpo.cuda()
           else:
               cnnpo = LeNet()
               cnnpo.load_state_dict(torch.load(args.MODELFOLDER + 'net_params_best.pth'))

           test_output_pNew = cnnpnew(test_x)
           test_output_porg = cnnpo(test_x)
           if use_gpu:
               #   pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
               predpNew_y = torch.max(test_output_pNew, 1)[1].cuda().data
               predporg_y = torch.max(test_output_porg, 1)[1].cuda().data
           else:
               predpNew_y = torch.max(test_output_pNew, 1)[1].data
               predporg_y = torch.max(test_output_porg, 1)[1].data
           accuracypNew = torch.sum(predpNew_y == test_y).type(torch.FloatTensor) / test_y.size(0)
           accuracyporg = torch.sum(predporg_y == test_y).type(torch.FloatTensor) / test_y.size(0)
           print('Epoch', epoch, '| new accuracy: %.5f' % accuracyNew, '| org accuracy: %.5f' % accuracyorg)
           if accuracypNew > accuracyporg:
               # 这样写有问题，就是会把new的训练参数删掉，需要进一步修改
               os.renames(args.MODELFOLDER + 'net_params_new.pth', args.MODELFOLDER + 'net_params_best.pth')
               print('best model save update in epoch', epoch)
           else:
               print('best model is still the original')


#加载模型和训练好的参数
def LeNetModerRestore():
    if use_gpu:
        cnnrestore = torch.load(args.MODELFOLDER+'cnn_entire_best.pth')
#        cnnrestore = torch.nn.DataParallel(cnnrestore, device_ids=range(torch.cuda.device_count()))
        cnnrestore = cnnrestore.cuda()
    else:
        cnnrestore = torch.load(args.MODELFOLDER+'cnn_entire_best.pth')
    test_output = cnnrestore(test_x)
    if use_gpu:
     #   pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
        pred_y = torch.max(test_output, 1)[1].cuda().data
    else:
        pred_y = torch.max(test_output, 1)[1].data
    print(pred_y)
    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print( 'test accuracy: %.2f' % accuracy)

#只加载训练好的参数
def LeNetParams():
    if use_gpu:
        cnnparams = LeNet()
        cnnparams.load_state_dict(torch.load(args.MODELFOLDER+'net_params_best.pth'))
#        cnnparams = torch.nn.DataParallel(cnnparams, device_ids=range(torch.cuda.device_count()))
        cnnparams = cnnparams.cuda()
    else:
        cnnparams = LeNet()
        cnnparams.load_state_dict(torch.load(args.MODELFOLDER+'net_params_best.pth'))
    test_output = cnnparams(test_x)
    if use_gpu:
      #  pred_y = torch.nn.DataParallel(pred_y, device_ids=range(torch.cuda.device_count()))
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
    LeNetModerRestore()
    LeNetParams()
