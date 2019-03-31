"""
code 설명
save is model
"""
from torch.utils.data.dataset import Dataset
import torch
from deep3_03_17 import Cnn_Model
from my_dataset import NkDataSet
from tensorboardX import SummaryWriter
import argparse
import time
import os
#argparse 커맨드로 부터 인자를 받아서 수행 할 수 있는 기능, fault로 기본값을 지정할 수 있따.

parser = argparse.ArgumentParser(description='PyTorch Coustom Training')
parser.add_argument('--print_freq', '--p', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save-import torch.nn.init as initdir',dest='save_dir',help='The directory used to save the trained models'
                    ,default='save_layer_load',type=str)
#metavar 는 mickname 같은 역할

#dest 입력되는 값이 저장되는 변수

args = parser.parse_args()

#save checkpoint

def save_checkpoint(state,filename='checkpoint.pth.bar'):

    torch.save(state,filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n # sum = sum + val * n
        self.count += n
        self.avg = self.sum / self.count

def train(my_dataset_loader,model,criterion,optimizer,epoch,writer):

    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model

        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data

        images = torch.autograd.Variable(images)
        label = torch.autograd.Variable(label)


        # 그냥 images를 하면 에러가 난다. 데이터 shape이 일치하지 않아서
        y_pred = model(images)

        #Compute and print loss
        loss = criterion(y_pred, label)

        #print (epoch, loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuracy(output.data, label)[0]

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    writer.add_scalar('Train/loss', losses.avg, epoch)
    writer.add_scalar('Train/accuaracy', top1.avg, epoch)

def test(my_dataset_loader, model, criterion, epoch, test_writer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    batch_time = AverageMeter()
    end = time.time()
    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model

        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data

        # 그냥 images를 하면 에러가 난다. 데이터 shape이 일치하지 않아서
        y_pred = model(images)

        # Compute and print loss
        loss = criterion(y_pred, label)

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuracy(output.data, label)[0]

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        #중간 과정을 보는 것
        if i % args.print_freq == 0:
            print('Test : [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i
                                                                  ,len(my_dataset_loader),batch_time=batch_time,loss=losses,top1=top1))

    print(' *, epoch : {epoch:.2f} Prec@1 {top1.avg:.3f}'.format(epoch=epoch,top1=top1))

    test_writer.add_scalar('Test/loss', losses.avg, epoch)
    test_writer.add_scalar('Test/accuaracy', top1.avg, epoch)

#Data_Load
csv_path = './test.csv'

custom_dataset = NkDataSet(csv_path)
my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=2,
                                                shuffle=False,
                                                num_workers=1)
model = Cnn_Model()

#CrossEntropyLoss 를 사용
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)#1/10000

writer = SummaryWriter('./log')
test_writer = SummaryWriter('./log/test')

args.save_dir = 'save_dir'
for epoch in range(500):
    train(my_dataset_loader,model,criterion,optimizer,epoch,writer)
    test(my_dataset_loader,model,criterion,epoch,test_writer)

    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict()
                     }, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
