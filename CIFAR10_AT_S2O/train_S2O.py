from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import copy

from models.preact_resnet import *

import time
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar10-resnet18/0.1',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')

#args = parser.parse_args()
args = parser.parse_args(args=[]) 

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    model.eval()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def train(args, model, device, train_loader, optimizer, epoch):
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            X, y = Variable(data, requires_grad=True), Variable(target)
            data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
        
        output = model(data)
        output_adv = model(data_adv)
        loss_adv = F.cross_entropy(output_adv, target)
        
        with torch.no_grad():
            aa = torch.zeros(output.size()[1], output.size()[1]).to(device)
            for xx in output:
                aa += torch.kron(xx, xx.reshape((-1,1)))
            #aa = torch.abs(torch.inverse(aa))
            aa = -1*torch.abs(aa)
            aa_min, aa_indexes = torch.min(aa, 1)
            aa = aa-aa_min.reshape(-1,1)
            for i in range(len(aa)):
                aa[i,i]=0.
            aa = 0.8*aa/aa.sum(1).reshape(-1,1)
            for i in range(len(aa)):
                aa[i,i]=0.2
            yy = aa[target]
            
            aa_adv = torch.zeros(output_adv.size()[1], output_adv.size()[1]).to(device)
            for xx in output_adv:
                aa_adv += torch.kron(xx, xx.reshape((-1,1)))
            #aa_adv = torch.abs(torch.inverse(aa_adv))
            aa_adv = -1*torch.abs(aa_adv)
            aa_adv_min, aa_adv_indexes = torch.min(aa_adv, 1)
            aa_adv = aa_adv-aa_adv_min.reshape(-1,1)
            for i in range(len(aa_adv)):
                aa_adv[i,i]=0.
            aa_adv = 0.8*aa_adv/aa_adv.sum(1).reshape(-1,1)
            for i in range(len(aa_adv)):
                aa_adv[i,i]=0.2
            yy_adv = aa_adv[target]
        
        train_loss += F.cross_entropy(output_adv, target, size_average=False).item()
        pred = output_adv.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss_yy = (1.0 / yy.size()[0]) * criterion_kl(F.log_softmax(model(data), dim=1),yy)
        loss_yy_adv = (1.0 / yy_adv.size()[0]) * criterion_kl(F.log_softmax(model(data_adv), dim=1),yy_adv)
        
        loss = 0.9*loss_adv + 0.1*0.5*(loss_yy+loss_yy_adv)
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    train_accuracy = correct / len(train_loader.dataset)
    return train_loss, train_accuracy

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    model = PreActResNet18(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    tstt = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        trnloss, trnacc = train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        tstloss, tstacc = eval_test(model, device, test_loader)
        
        print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tstacc))
        #print('test_adv_loss: {:.4f}, test_adv_acc: {:.2f}%'.format(tst_adv_loss, 100. * tst_adv_acc))
        
        tstt.append(tstacc)
        # save checkpoint
        if (epoch>99 and epoch%5==0) or (epoch>99 and epoch<110) or (epoch>149 and epoch<160) or (epoch>99 and tstacc==max(tstt)):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))

if __name__ == '__main__':
    main()
