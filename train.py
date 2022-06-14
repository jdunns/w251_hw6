from pickle import TRUE
from tabnanny import verbose
import os
import pickle
import random
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# Constants
ARCHITECTURE_CLASS = torchvision.models.resnet18
CRITERION_CLASS = torch.nn.CrossEntropyLoss
DATA_DIR = './data/'
EPOCHS = 100
IMG_SIZE = 224
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 1000
NUM_WORKERS = 8
OPTIMIZER_CLASS = torch.optim.SGD
PRINT_FREQ = 50
SEED = 1
START_EPOCH = 0
TRAIN_BATCH_SIZE = 420
VAL_BATCH_SIZE = 420
WEIGHT_DECAY = 4.5e-5

# Check CUDA availability.
print(f'CUDA Available:  {torch.cuda.is_available()}')
print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
DEVICE = 'cuda'

# Set certain runtime settings
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

# enable algorithm optimization
cudnn.benchmark = True

def train(train_loader, model:ARCHITECTURE_CLASS,
          criterion:CRITERION_CLASS,
          optimizer:OPTIMIZER_CLASS,
          epoch):
    batch_time_meter = AverageMeter('Time', ':6.3f')
    data_time_meter = AverageMeter('Data', ':6.3f')
    losses_meter = AverageMeter('Loss', ':.4e')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')
    progress_meter = ProgressMeter(
        len(train_loader),
        [batch_time_meter, data_time_meter, losses_meter, top1_meter, top5_meter],
        prefix="Epoch: [{}]".format(epoch))

    ######################
    # switch model to train mode here
    model.train()
    ################

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time_meter.update(time.time() - end)

        #####################
        # send the images to cuda device
        # send the target to cuda device
        images = images.to(device=DEVICE)
        target = target.to(device=DEVICE)


        # compute output
        output = model(images)

        # compute loss 
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1[0], images.size(0))
        top5_meter.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        
        #### zero out gradients in the optimizer
        optimizer.zero_grad(set_to_none=True)
        
        ## backprop!
        loss.backward()
        
        # update the weights!
        optimizer.step()

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            progress_meter.display(i)

def validate(val_loader: torch.utils.data.DataLoader,
             model: ARCHITECTURE_CLASS,
             criterion: CRITERION_CLASS):
    batch_time_meter = AverageMeter('Time', ':6.3f')
    losses_meter = AverageMeter('Loss', ':.4e')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')
    progress_meter = ProgressMeter(
        len(val_loader),
        [batch_time_meter, losses_meter, top1_meter, top5_meter],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            ### send the images and target to cuda
            images = images.to(device=DEVICE)
            target = target.to(device=DEVICE)
            # compute output
            output = model(images)

            # compute loss 
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1[0], images.size(0))
            top5_meter.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                progress_meter.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1_meter, top5=top5_meter))

    return top1_meter.avg

def save_checkpoint(state, is_best, filename='./.checkpoint.pkl'):
    # save the model state!
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    
    if is_best:
        shutil.copyfile(filename, './.best_model.pkl')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    # This *might* speed things up
    torch.backends.cudnn.benchmark = True

    # select the model
    model = ARCHITECTURE_CLASS()
    
    # Make sure the output layer has the correct number of features.
    model.fc = torch.nn.Linear(512, NUM_CLASSES)

    # use the cross-entropy loss
    criterion = CRITERION_CLASS()

    # use SGD .. use the momentum and weight decay vars
    optimizer = OPTIMIZER_CLASS(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    # Setup the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=11,
                                                gamma=0.1,
                                                verbose=True)

    # Training data
    transform_train = transforms.Compose([
        #transforms.AutoAugment(),
        transforms.RandomResizedCrop(size=IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageNet(DATA_DIR, split='train', transform=transform_train)
    # train_dataset = torchvision.datasets.CIFAR10('./.temp/train/', train=True, download=True, transform=transform_train)

    # Validation data
    transform_val = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    val_dataset = torchvision.datasets.ImageNet(DATA_DIR, split='val', transform=transform_val)
    # val_dataset = torchvision.datasets.CIFAR10('./.temp/test/', train=False, download=True, transform=transform_val)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # send the model to the cuda device.. 
    model.to(device=DEVICE)

    # Load the checkpoint if it exists
    if os.path.exists('./.checkpoint.pkl'):
        with open('./.checkpoint.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
            START_EPOCH = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    best_acc1 = 0
    for epoch in range(START_EPOCH, EPOCHS):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch,
            'architecture': ARCHITECTURE_CLASS.__name__,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        scheduler.step()
        print('Learning Rate: ' + str(scheduler.get_last_lr()))
        print(f'\n Validation Top1 Accuracy: {acc1}\n\n')