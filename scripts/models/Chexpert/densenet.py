import torch.nn as nn
import torchvision
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from config import opt

__all__ = ['densenet121', 'densenet169', 'densenet201']


def densenet121(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet121(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def densenet169(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet169(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


def densenet201(num_classes, pretrained=False, **kwargs):
    model = torchvision.models.densenet201(pretrained=pretrained, **kwargs)

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()

    )
    return model
'''
def _val(model, dataloader, criterion, total_batch):
    model.eval()
    counter = 0
    loss_sum = 0

    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, (data, label) in bar:
            inp = data.clone().detach()
            target = label.clone().detach()
            if opt.use_gpu:
                inp = inp.cuda()
                target = target.cuda()

            output = model(inp)

            loss = criterion(output, target)
            loss_sum += loss.item()
            counter += 1
            bar.set_postfix_str('loss: %.5s' % loss.item())

    loss_mean = loss_sum / counter
    return loss_mean
'''
def _val(model, dataloader, criterion, total_batch):
    model.eval()
    counter = 0
    loss_sum = 0

    OUTPUTS = []
    LABELS = []
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, (data, label) in bar:
            inp = data.clone().detach()
            target = label.clone().detach()
            if opt.use_gpu:
                inp = inp.cuda()
                target = target.cuda()

            output = model(inp).detach().cpu().numpy()
            OUTPUTS.extend(output)
            LABELS.extend(label.detach().cpu().numpy())
    LABELS = np.asarray(LABELS)
    OUTPUTS = np.asarray(OUTPUTS)
    
    prc = 0
    for i in range(LABELS.shape[1]-1):
        p, r, t = precision_recall_curve(LABELS[:, i], OUTPUTS[:, i])
        prc += auc(r, p)
    # aucroc
    #roc = sum([roc_auc_score(LABELS[:, i], OUTPUTS[:, i]) for i in range(LABELS.shape[1]-1)])
    prc = prc / (LABELS.shape[1] -1)
    bar.set_postfix_str('auprc: %.5s' % prc)
    print(prc)
	

    # loss_mean = prc / (LABELS.shape[0]-1)
    # return loss_mean
    return prc


def train(model, train_dataloader, val_dataloader, use_gpu=True):

    if use_gpu:
        model = model.to(opt.device)
        model = nn.DataParallel(model)

    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas,
                                 eps=opt.eps, weight_decay=opt.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    # step4: meters
    max_auprc = 0
    
    # train
    print('\n---------------------------------')
    print(' ( ᐛ )و - Start training ......')
    print('---------------------------------\n')
    for epoch in range(opt.max_epoch):
        print('(ㅂ)و - Epoch', epoch + 1)
        model.train()
        total_batch = int(train_dataloader.__len__ ()/ opt.bs)

        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        for i, (data, label) in bar:
            # train model
            torch.set_grad_enabled(True)
            inp = data.clone().detach().requires_grad_(True)
            target = label.clone().detach()
            if use_gpu:
                inp = inp.cuda()
                target = target.cuda()
			
            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        auprc = _val(model, val_dataloader, criterion, total_batch)
        time_end = time.strftime('%m%d_%H%M%S')
        scheduler.step(-auprc)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "./checkpoints/epoch_{}.pth".format(epoch+1))
        if max_auprc < auprc:
            max_auprc = auprc
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       './checkpoints/m_' + time_end + '.pth.tar')
            print('(ᗜ) Epoch [' + str(epoch + 1) + '] [save] [m_' + time_end + '] auprc= ' + str(auprc))
        else:
            print('(இωஇ) Epoch [' + str(epoch + 1) + '] [----] [m_' + time_end + '] auprc= ' + str(auprc))
        print('----------------------------------------------------------------------\n')

if __name__ == '__main__':
    # For Mac: run '/Applications/Python\ 3.7/Install\ Certificates.command' is necessary
    assert densenet121(num_classes=10, pretrained=True)
    print('success')
    assert densenet169(num_classes=10, pretrained=True)
    print('success')
    assert densenet201(num_classes=10, pretrained=True)
    print('success')
