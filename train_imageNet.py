# %%
import argparse
import shutil
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


# %%
def load_imagenet_data(data_dir, batch_size, num_worker):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_trainsform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load ImageNet training and validation datasets
    train_dataset = datasets.ImageNet(root=data_dir, split='train', transform=train_transform)
    val_dataset = datasets.ImageNet(root=data_dir, split='val', transform=val_trainsform)
    num_classes = len(train_dataset.classes)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory = True)

    return train_loader, val_loader, num_classes

def load_cifar10_data(data_dir, batch_size, num_worker):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)
    num_classes = len(train_dataset.classes)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)
    return train_loader, test_loader, num_classes


# %%
#import timm

def initialize_model(model_name, num_classes, pretrained = False):
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    if model_name in model_names:
        if pretrained:
            print(f"using pretrained model {pretrained}")
            model = models.__dict__[model_name](weights = pretrained)
            assert model.fc.out_features == num_classes, "pretrained num_class is not matched with dataset"
        else:
            model = models.__dict__[model_name]()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    # elif model_name == 'vit_base_patch16_224':
    #     model = timm.create_model('vit_base_patch16_224', pretrained=use_pretrained, num_classes=num_classes)
    else:
        raise Exception("Model not supported: {}".format(model_name))

    return model

# %%
def train(train_loader, model, criterion, optimizer, device, epoch):
    # train for one epoch
    data_time = AverageMeter('Data_Time', ':6.3f')
    batch_time = AverageMeter('Batch_Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    metrics_list = [losses, top1, top5, data_time, batch_time, ]
    
    model.train() # switch to train mode

    end = time.time()

    tqdm_epoch = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}', total=len(train_loader))
    for images, target in tqdm_epoch:
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        tqdm_epoch.set_postfix(train_metrics = ", ".join([str(x) for x in metrics_list]))

        end = time.time()
    tqdm_epoch.close()

    writer.add_scalar(f'Train Loss', losses.avg, epoch)
    writer.add_scalar(f'Train Accuracy@1', top1.avg, epoch)
    writer.add_scalar(f'Train Accuracy@5', top5.avg, epoch)


def validate(val_loader, model, criterion, device, epoch = 0):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    metrics_list = [losses, top1, top5]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        tqdm_val = tqdm(val_loader, desc='Validation', total=len(val_loader))
        for images, target in tqdm_val:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            tqdm_val.set_postfix(val_metrics = ", ".join([str(x) for x in metrics_list]))

        tqdm_val.close()
    
    writer.add_scalar(f'Val Loss', losses.avg, epoch)
    writer.add_scalar(f'Val Accuracy@1', top1.avg, epoch)
    writer.add_scalar(f'Val Accuracy@5', top5.avg, epoch)

    return top1.avg

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
        fmtstr = 'avg {name}: {avg' + self.fmt + '} (n={count}))'
        return fmtstr.format(**self.__dict__)

def calculate_accuracy(output, target, topk=(1,)):
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

def save_checkpoint(state, is_best, filepath):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'best_model.pth'))


# %%
best_acc1 = 0
writer = SummaryWriter()
print("writing tensorboard result to ./runs/")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = '/datasets/ILSVRC2012' 
    model_architecture = "resnet50"
    pretrained = False #"IMAGENET1K_V2"
    start_epoch = 0
    num_epochs = 90
    batch_size = 128
    num_worker = 4
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    from_checkpoint = False #"checkpoints/checkpoint.pth" 
    evaluate_mode = False

    global best_acc1
    print(f"using '{device}'")
    print(f"using model '{model_architecture}'")

    train_loader, val_loader, num_classes = load_imagenet_data(data_dir, batch_size = batch_size, num_worker = num_worker)
    # data_dir = '/home/jinhyun/datasets/CIFAR10' 
    # train_loader, val_loader, num_classes = load_cifar10_data(data_dir, batch_size = batch_size, num_worker = num_worker)

    model = initialize_model(model_architecture, num_classes = num_classes, pretrained = pretrained) 
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # learning rate to the initial LR decayed by 10 every 30 epochs

    if from_checkpoint:
        if os.path.isfile(from_checkpoint):
            print("=> loading checkpoint '{}'".format(from_checkpoint))
            checkpoint = torch.load(from_checkpoint, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint from file '{from_checkpoint}' (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found at '{}'".format(from_checkpoint))

    if evaluate_mode:
        acc1 = validate(val_loader, model, criterion, device)
        return
    
    for epoch in range(start_epoch, num_epochs):
        train(train_loader, model, criterion, optimizer, device, epoch)
        acc1 = validate(val_loader, model, criterion, device, epoch)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, filepath = "checkpoints/checkpoint.pth")



# %%
main()
writer.close()


# %%
def test_data_load_time(data_loader):
    import time

    # Number of batches to read
    num_batches_to_read = 100

    start_time = time.time()
    for i, batch in enumerate(data_loader):
        if i >= num_batches_to_read:
            break
        # Optionally, process the batch here if needed
    end_time = time.time()
    print(end_time - start_time)


