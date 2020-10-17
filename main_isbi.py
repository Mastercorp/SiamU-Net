from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data.dataloader as dl
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sacred import Experiment
from sacred.observers import MongoObserver
from module import spidermodel as md, ISBI2012Data_2 as ISBI
import time
import sys

from sacred.settings import SETTINGS

SETTINGS.DISCOVER_SOURCES = 'sys'

# use sacred to handle experiment parameters
ex = Experiment("15 Double_relu r d f ob 1")
ex.add_config('config.json')


@ex.config
def my_config(params):
    if not os.path.exists(params["savedir"]):
        os.makedirs(str(params["savedir"]))
    elif not params["resume"]:
        dirindex = 1
        while os.path.exists(params["savedir"][:-1] + str(dirindex) + "/"):
            dirindex += 1
        params["savedir"] = params["savedir"][:-1] + str(dirindex) + "/"
        os.makedirs(str(params["savedir"]))
    else:
        params["savedir"] = params["resume"][:params["resume"].rfind("/") + 1]

    if not params["evaluate"]:
        pass
        # mongourl = ("connect with your mongodb")
        # ex.observers.append(MongoObserver.create(url=mongourl, db_name='isbidb'))


# Main code:

torch.backends.cudnn.deterministic = True
# check if cuda is available
udevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.capture
def save_checkpoint(state, filename, params):
    torch.save(state, str(params["savedir"]) + filename)


def save_images(outputs, filepath, filename, classes):
    # copy first image in outputs back to cpu and save it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    for i in range(int(classes)):
        img = Image.fromarray((outputs[0][i][:][:].cpu().detach().numpy() * 255).astype(np.uint8))
        img.save(filepath + filename.format(i) + '.tif')


def save_images_eval(outputs, filepath, filename, classes):
    # copy first image in outputs back to cpu and save it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    for i in range(int(classes)):
        img = Image.fromarray((outputs[0][i][:][:].cpu().detach().numpy()))
        img.save(filepath + filename.format(i) + '.tif')


def iou_loss(outputs, label):
    epsi = 1e-6

    outputs[outputs < 0.5] = 0
    outputs[outputs >= 0.5] = 1

    tp = torch.mul(outputs, label).sum((1, 2))
    tn = torch.mul(outputs.add(-1).abs(), label.add(-1).abs()).sum((1, 2))
    fp = torch.mul(outputs, label.add(-1).abs()).sum((1, 2))
    fn = torch.mul(outputs.add(-1).abs(), label).sum((1, 2))

    precision = ((tp / (tp + fp + epsi)).sum() / outputs.shape[0]).item()
    fpr = ((fp / (fp + tn + epsi)).sum() / outputs.shape[0]).item()
    recall = ((tp / (tp + fn)).sum() / outputs.shape[0]).item()
    accuracy = (((tp + tn) / (tp + tn + fp + fn)).sum() / outputs.shape[0]).item()
    f1score = (((2 * tp) / (2 * tp + fp + fn)).sum() / outputs.shape[0]).item()

    iou = (((tp + epsi) / (fp + tp + fn + epsi)).sum() / outputs.shape[0]).item()
    return iou, precision, recall, accuracy, f1score, fpr


@ex.capture
def train(trainloader, model, criterion, optimizer, epoch, params):
    model.train()
    loss_sum = 0
    iou_loss_sum = (0, 0, 0, 0, 0, 0)
    optimizer.zero_grad()
    for i, (trainimg, label, imgname) in enumerate(trainloader):
        # get train and label data
        # the first return value, which is an index.
        # put on gpu or cpu
        trainimg = trainimg.to(udevice)
        label = label.to(udevice)

        label = label.view(label.size(0), label.size(2), label.size(3))

        # forward + backward + optimize
        outputs = model(trainimg, padding=params["padding"])

        # # to save every image, just remove the "and (i == 0)" part
        if params["saveimages"] and (i == 0):
            save_images(outputs, str(params["savedir"] + "output/"), str("class_{}_" + str(epoch) + "_image_" + str(i)),
                        int(params["classes"]))
        outputs = outputs.view(outputs.size(0), outputs.size(2), outputs.size(3))
        loss = criterion(outputs, label)

        # to use classweights with BCE one has to create its own weights
        weight = torch.tensor(params["classweight"]).to(udevice)
        weight_ = weight[label.data.view(-1).long()].view_as(label)
        loss_class_weighted = loss * weight_.float()

        loss_class_weighted = loss_class_weighted.mean()
        loss_class_weighted.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss = loss_class_weighted.item()
        loss_sum = loss_sum + running_loss
        iou_loss_sum = tuple(sum(x) for x in zip(iou_loss(outputs, label), iou_loss_sum))
        # delete all references to variables:
        del outputs, trainimg, label, loss_class_weighted, loss

    loss_avg = loss_sum / (i + 1)
    loss_iou_avg = tuple(x / (i + 1) for x in iou_loss_sum)

    return loss_avg, loss_iou_avg


# torch.no_grad() is the same as adding with torch.no_grad(): for the whole evaluation block
@ex.capture
@torch.no_grad()
def evaluate(valloader, model, criterion, epoch, params):
    # switch the model to eval mode ( important for dropout layers or batchnorm layers )
    model.eval()
    loss_sum = 0
    iou_loss_sum = (0, 0, 0, 0, 0, 0)
    for i, data in enumerate(valloader):
        # get train and label data
        val, label, imgname = data
        name = str(imgname[0])

        val = val.to(udevice)
        label = label.to(udevice)

        label = label.view(label.size(0), label.size(2), label.size(3))
        # forward + backward + optimize
        outputs = model(val, padding=params["padding"])

        if not params["evaluate"]:
            save_images(outputs, str(params["savedir"] + "val/"), str(name + "_" + str(epoch)),
                        int(params["classes"]))
        else:
            save_images_eval(outputs, str(params["savedir"] + "eval" + str(epoch) + "/"), str(name),
                             int(params["classes"]))

        outputs = outputs.view(outputs.size(0), outputs.size(2), outputs.size(3))
        loss = criterion(outputs, label)
        iou_loss_sum = tuple(sum(x) for x in zip(iou_loss(outputs, label), iou_loss_sum))
        weight = torch.tensor(params["classweight"]).to(udevice)
        weight_ = weight[label.data.view(-1).long()].view_as(label)
        loss_class_weighted = loss * weight_.float()
        loss_class_weighted = loss_class_weighted.mean()

        running_loss = loss_class_weighted.item()
        loss_sum = loss_sum + running_loss
        del outputs, val, label, loss

    loss_avg = loss_sum / (i + 1)
    loss_iou_avg = tuple(x / (i + 1) for x in iou_loss_sum)
    return loss_avg, loss_iou_avg


@ex.automain
def my_main(params, augment):
    print("***** Starting Programm *****")
    # initialize some variables
    train_loss = []
    iou_train_loss = []
    val_loss = []
    iou_val_loss = []
    best_loss = 10
    best_iou = 0

    global udevice

    if params["cpu"]:
        udevice = torch.device("cpu")

    # use cudnn for better speed, if available
    if udevice.type == "cuda":
        cudnn.benchmark = True

    # 1 Model
    model = md.DanNet().to(udevice)

    # 2 Construct loss and optimizer
    # Using a softmax layer at the end, applying the log and using NLLoss()
    # has the same loss as using no softmax layer, and calculating the CrossEntropyLoss()
    # the difference is in the output image of the model.
    # If you want to use the CrossEntropyLoss(), remove the softmax layer, and  the torch.log() at the loss

    # class weights can be used with NLLLoss

    # need sigmoid at the end layer, to calculate iou loss
    # use reduction none to generate own weights in train
    criterion = nn.BCELoss(reduction='none').to(udevice)
    optimizer = optim.SGD(model.parameters(), lr=params["learningrate"],
                          momentum=params["momentum"], weight_decay=params["weightdecay"])

    trainset = ISBI.ISBIDataset(
        "./ISBI 2012/Train-Volume/train-volume-*.tif", "./ISBI 2012/Train-Labels/train-labels-*.tif",
        eval=False, augment=augment)

    valset = ISBI.ISBIDataset(
        "./ISBI 2012/Val-Volume/train-volume-*.tif", "./ISBI 2012/Val-Labels/train-labels-*.tif",
        eval=True, augment=augment)

    chelset = ISBI.ISBIDataset(
        "./ISBI 2012/Test-Volume/test-volume-*.tif", "./ISBI 2012/Test-Volume/test-volume-*.tif",
        eval=True, augment=augment)

    trainloader = dl.DataLoader(trainset, shuffle=True, num_workers=params["workers"], batch_size=params["batch_size"],
                                drop_last=True, pin_memory=True)
    valloader = dl.DataLoader(valset, num_workers=2, batch_size=1, pin_memory=True)
    chelloader = dl.DataLoader(chelset, num_workers=2, batch_size=1, pin_memory=True)

    # 3 Training cycle forward, backward , update

    # load the model if set
    startepoch = 0
    if params["resume"]:
        if os.path.isfile(params["resume"]):
            print("=> loading checkpoint '{}'".format(params["resume"]))
            checkpoint = torch.load(params["resume"])
            startepoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            iou_train_loss = checkpoint['iou_train_loss']
            iou_val_loss = checkpoint['iou_val_loss']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params["resume"], checkpoint['epoch']))
            # only if you load a model, you can use predict
            if params["evaluate"]:
                valloss, iouvalloss = evaluate(chelloader, model, criterion, startepoch)

        else:
            print("=> no checkpoint found at '{}'".format(params["resume"]))
            sys.exit(0)

    if params["txtinfo"] and not params["evaluate"]:
        # print some info for console
        print('Dataset      :  ISBI 2012')
        print('Start Epoch  : ' + str(startepoch))
        print('End Epoch    : ' + str(params["epochs"]))
        print('Learning rate: ' + str(params["learningrate"]))
        print('Momentum     : ' + str(params["momentum"]))
        print('Weight decay : ' + str(params["weightdecay"]))
        print('Use padding  : ' + str(params["padding"]))
        print('Use classweights  : ' + str(params["classweight"]))

        with open(str(params["savedir"]) + "txtinfo.txt", "a") as myfile:
            myfile.write('Dataset      : ISBI 2012')
            myfile.write('\n')
            myfile.write('Start Epoch  : ' + str(startepoch))
            myfile.write('\n')
            myfile.write('End Epoch    : ' + str(params["epochs"]))
            myfile.write('\n')
            myfile.write('Learning rate: ' + str(params["learningrate"]))
            myfile.write('\n')
            myfile.write('Momentum     : ' + str(params["momentum"]))
            myfile.write('\n')
            myfile.write('Weight decay : ' + str(params["weightdecay"]))
            myfile.write('\n')
            myfile.write('Use padding  : ' + str(params["padding"]))
            myfile.write('\n')
            myfile.write('Use classweights  : ' + str(params["classweight"]))
            myfile.write('\n')
            myfile.close()

        print("***** Start Training *****")
        breakloss = 0
        for epoch in range(startepoch, params["epochs"]):
            start_time = time.time()

            trainloss, ioutrainloss = train(trainloader, model, criterion, optimizer, epoch + 1)
            train_loss.append(trainloss)
            iou_train_loss.append(ioutrainloss)
            valloss, iouvalloss = evaluate(valloader, model, criterion, epoch + 1)
            val_loss.append(valloss)
            iou_val_loss.append(iouvalloss)
            end_time = time.time()
            out_str = "Epoch [{:4d}] train_loss: {:.4f} val_loss: {:.4f}" \
                      " tr_iou: {tx[0]:.4f} tprec: {tx[1]:.4f} tfpr: {tx[5]:.4f} tre: {tx[2]:.4f} tacc: {tx[3]:.4f} tf1: {tx[4]:.4f}" \
                      " va_iou: {vx[0]:.4f} vprec: {vx[1]:.4f} vfpr: {vx[5]:.4f} vre: {vx[2]:.4f} vacc: {vx[3]:.4f} vf1: {vx[4]:.4f}" \
                      " loop time: {looptime:.4f}"
            print(out_str.format(epoch + 1, train_loss[epoch], val_loss[epoch], tx=iou_train_loss[epoch],
                                 vx=iou_val_loss[epoch], looptime=(end_time - start_time)))
            if params["txtinfo"]:
                with open(str(params["savedir"]) + "txtinfo.txt", "a") as myfile:
                    myfile.write(out_str.format(epoch + 1, train_loss[epoch], val_loss[epoch], tx=iou_train_loss[epoch],
                                                vx=iou_val_loss[epoch], looptime=(end_time - start_time)))
                    myfile.write('\n')
                    myfile.close()

            if 0.6931 < train_loss[epoch] < 0.6932:
                breakloss += 1
                if breakloss > 7:
                    print("does not converge")
                    sys.exit()
            else:
                breakloss = 0

            # save best loss
            is_best_loss = val_loss[epoch] < best_loss
            best_loss = min(val_loss[epoch], best_loss)

            if is_best_loss:
                ex.log_scalar('best_epoch', epoch + 1)

            is_best_iou = iou_val_loss[epoch][0] > best_iou
            best_iou = max(iou_val_loss[epoch][0], best_iou)

            if is_best_iou:
                ex.log_scalar('best_iou_epoch', epoch + 1)

            ex.log_scalar('val_loss', val_loss[epoch])
            ex.log_scalar('train_loss', train_loss[epoch])
            ex.log_scalar('iou_val_loss', iou_val_loss[epoch][0])
            ex.log_scalar('iou_train_loss', iou_train_loss[epoch][0])
            ex.log_scalar('vprec', iou_val_loss[epoch][1])
            ex.log_scalar('tprec', iou_train_loss[epoch][1])
            ex.log_scalar('vre', iou_val_loss[epoch][2])
            ex.log_scalar('tre', iou_train_loss[epoch][2])
            ex.log_scalar('vacc', iou_val_loss[epoch][3])
            ex.log_scalar('tacc', iou_train_loss[epoch][3])
            ex.log_scalar('vf1', iou_val_loss[epoch][4])
            ex.log_scalar('tf1', iou_train_loss[epoch][4])
            ex.log_scalar('vfpr', iou_val_loss[epoch][5])
            ex.log_scalar('tfpr', iou_train_loss[epoch][5])

            # save model
            filename = ""
            if is_best_loss:
                filename = 'best_loss.pth.tar'
            else:
                filename = 'current.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'iou_train_loss': iou_train_loss,
                'iou_val_loss': iou_val_loss,
                'optimizer': optimizer.state_dict(),
            }, filename=filename)

            if is_best_iou:
                filename = 'best_iou_loss.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'iou_train_loss': iou_train_loss,
                    'iou_val_loss': iou_val_loss,
                    'optimizer': optimizer.state_dict(),
                }, filename=filename)

        print("*****   End  Programm   *****")
