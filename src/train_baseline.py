# import libraries
import argparse
import shutil
import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from sklearn import metrics
from timeit import default_timer
import seed
from sms import send
import config
# import pdb; pdb.set_trace();


def parse_arguments():
    """ Parse arguments """

    parser = argparse.ArgumentParser()

    # task
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])

    # plane
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])

    # experiment's prefix name
    parser.add_argument('--prefix_name', type=str, required=True)

    # model
    parser.add_argument('--model', type=str, default='pretrained-resnet18')

    # epochs
    parser.add_argument('--epochs', type=int, default=100)

    # Image Slicing
    # Way slices are made (new planes are created this way)
    # vertical (default), horizontal, diagonal
    parser.add_argument('--cut', type=str, default='vertical')

    # augmenting images
    parser.add_argument('--augment', type=str, default='albumentations-group')

    # augmenting probability
    parser.add_argument('--augment_prob', type=float, default=0.5)

    # # patience
    # parser.add_argument('--patience', type=int, default=5)

    # LR scheduler
    parser.add_argument('--lr_scheduler', type=str,
                    choices=['plateau', 'step'], default='plateau')

    # # remove history
    # parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)

    # save model
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)

    # logging
    parser.add_argument('--log_every', type=int, default=100)

    args = parser.parse_args()
    return args

def collate_fn(batch):
    # This keeps the varying-sized 3D volumes separate in a list rather than stacking them
    return tuple(zip(*batch))

def run(args):
    """ Run training """

    ## START

    start = default_timer()

    # DEVICE

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LOGS

    # Create a folder that'll be used by tensorboard 
    # to save the training logs and visualize the metrics of the training session
    # On each run of the script, a new folder (named after the timestamp) is created:
    
    log_root_folder = "./logs/"

    # if args.flush_history == 1:
    #     objects = os.listdir(log_root_folder)
    #     for f in objects:
    #         if os.path.isdir(log_root_folder + f):
    #             shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = os.path.join(log_root_folder, '{}_{}_{}'.format(
        now.strftime("%Y%m%d-%H%M%S"), args.task, args.plane))

    print('[TRAIN] Creating logs folder: "{}"'.format(logdir))

    # Create dir
    os.makedirs(logdir)

    MAIN_LOG = os.path.join(logdir, 'mainlog.txt')

    with open(MAIN_LOG, 'w') as f:
        print("""MRNET 
Experiment: {}
TASK: {}
PLANE: {}
Prefix: {}
Model: {}
Save Model: {}
Device: {}
Epochs: {}
Augmentation: {}
LR: {}""".format(
    now.strftime("%d%m%Y-%H%M%S"), args.task, args.plane, args.prefix_name, args.model, args.save_model, 
    device, args.epochs, args.augment, config.LEARNING_RATE), file=f)

    # args.lr_scheduler
    # args.patience

    ## TENSORBOARD

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(logdir)

    ## DATA LOADER

    print('[TRAIN] Loading Data Loaders')

    from loaders.loader_baseline import MRDataset

    ## AUGMENTATION

    augment = args.augment
    augment_prob = args.augment_prob

    ## CUT

    cut = args.cut
    
    # Instantiate a train and validation MRDataset(s)

    train_dataset = MRDataset(args.task, args.plane, train_val_test='train', cut=cut, 
        augment=augment, augment_prob=augment_prob)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, drop_last=False, collate_fn=collate_fn)

    validation_dataset = MRDataset(args.task, args.plane, train_val_test='valid', cut=cut, 
        augment=augment, augment_prob=augment_prob)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=8, shuffle=False, drop_last=False, collate_fn=collate_fn)

    # MODEL

    print('[TRAIN] Instantiate Model')

    from models import choose

    # Get model
    mrnet = choose.get_model(args.model)
    mrnet.to(device)

    # Get model's architecture's name irrespective whether it's a model we are loading
    model_name = args.model
    if 'arch_' in model_name:
        model_name = model_name.split('arch_')[1].split('_')[0]

    # OPTIMIZER

    print('[TRAIN] Defining Optimizer')

    # Adam optimizer as well as a learning rate scheduler

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.1

    optimizer = optim.Adam(
        mrnet.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=3, 
        factor=.3, 
        threshold=1e-4, 
        )

    best_val_loss = float('inf')
    best_val_auc = float(0)

    ## TRAIN & EVALUATE

    print('[TRAIN] Starting Training!')

    num_epochs = args.epochs
    iteration_change_loss = 0
    # patience = args.patience
    log_every = args.log_every

    print('[TRAIN] Epochs = {}'.format(num_epochs))

    t_start_training = time.time()

    for epoch in range(num_epochs):

        print('[TRAIN] EPOCH # {}'.format(epoch + 1))

        # Learning Rate

        current_lr = get_lr(optimizer)

        # Start timer

        t_start = time.time()

        # TRAIN

        train_loss, train_auc = train_model(
            mrnet, train_loader, device, epoch, num_epochs, optimizer, writer, 
            MAIN_LOG, current_lr, log_every)
        
        # EVALUATE

        val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, device, epoch, num_epochs, writer, 
            MAIN_LOG, current_lr)

        # Scheduler

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        # End timer

        t_end = time.time()
        delta = (t_end - t_start) / 60.

        # Print
        _statement = 'Epoch {}:\nTrain loss: {:.4f} | Train AUC: {:.4f}\nVal Loss: {:.4f} | Val AUC: {:.4f}\nElapsed time: {:.2f} min'.format(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, delta)
        if val_auc > best_val_auc: _statement += '\nModel will be saved'
        with open(MAIN_LOG, 'a') as f: print(_statement, file=f)
        # Send
        with open(MAIN_LOG) as f:
            text = f.read()
        send(text)

        # LEARNING RATE Update?

        # if args.lr_scheduler == 1:
        #     print('[RUN][INFO] Updating learning rate..')
        #     scheduler.step(val_loss)

        # iteration_change_loss += 1

        print('-' * 100)

        # SAVE MODEL

        if val_auc > best_val_auc:
            # Update best
            best_val_auc = val_auc
            if bool(args.save_model):
                # Remove previous models
                # for f in os.listdir('./models/'):
                #     if args.task in f and args.plane in f and args.prefix_name in f:
                #         os.remove('./models/{}'.format(f))
                # Save
                model_filename = '{}/model_{}_{}_{}_train_auc_{:.4f}_val_auc_{:.4f}_train_loss_{:.4f}_val_loss_{:.4f}_epoch_{}_arch_{}_cut_{}_augment_{}_augment-probability_{}.pth'.format(
                    config.TRAIN_MODELS_PATH_PRETRAINED, args.prefix_name, args.task, args.plane, 
                    train_auc, val_auc, train_loss, val_loss, epoch + 1, model_name, 
                    cut, augment, augment_prob)
                torch.save(mrnet.state_dict(), model_filename)
                _statement = 'Model saved: "{}"'.format(model_filename)
                print(_statement)

        # PATIENCE

        # if val_loss < best_val_loss:
        #     # Save best val loss
        #     best_val_loss = val_loss
        #     iteration_change_loss = 0

        # if iteration_change_loss == patience:
        #     _statement = 'Early stopping after {0} iterations without the decrease of the val loss'.format(iteration_change_loss)
        #     print(_statement)
        #     with open(MAIN_LOG, 'a') as f: print(_statement, file=f)
        #     # Send
        #     with open(MAIN_LOG) as f:
        #         text = f.read()
        #     send(text)
        #     break

    ## END

    end = default_timer() - start
    minutes, seconds = divmod(end, 60)
    _statement = '[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds)
    print(_statement)
    with open(MAIN_LOG, 'a') as f: print(_statement, file=f)

    ## SEND TEXT

    with open(MAIN_LOG) as f:
        text = f.read()
    send(text)


def train_model(model, train_loader, device, epoch, num_epochs, optimizer, writer, log_file, current_lr, log_every=100):
    print('[TRAIN] Train model')
    _ = model.train()
    torch.cuda.empty_cache()

    scaler = torch.cuda.amp.GradScaler()

    y_preds = []
    y_trues = []
    losses = []
    total_step = len(train_loader)

    for i, (indices, images, labels, weights) in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_loss_val = 0
        for j in range(len(images)):
            image = images[j].to(device)
            label = labels[j].to(device)
            weight = weights[j].to(device)

            prediction = model.forward(image.float())[0]

            loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
            
            # Impartim loss-ul la numarul de poze pentru a pastra echilibrul matematic
            loss = loss / len(images)
            
            # [SECRETUL ESTE AICI] - Facem backward IN INTERIORUL buclei!
            # Asta sterge instantaneu istoricul memoriei, ramanand doar gradientii utiles.
            loss.backward()
            
            # Salvam doar numarul pur pentru afisare (fara graficul din memorie)
            batch_loss_val += loss.item() * len(images)
            
            y_pred = torch.sigmoid(prediction).item()
            y_true = int(label.item())
            y_preds.append(y_pred)
            y_trues.append(y_true)

        # Facem pasul optimizatorului abia dupa ce am acumulat gradientii celor 8 pacienti
        optimizer.step()
        
        loss_value = batch_loss_val / len(images)
        losses.append(loss_value)

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value, epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if i > 0 and i % log_every == 0:
            _statement = """Epoch [{0} / {1}] Step: [{2} / {3}] avg_train_loss: {4} | train_auc: {5} | lr : {6}""".format(
                    epoch + 1, num_epochs, i, total_step,
                    np.round(np.mean(losses), 4), np.round(auc, 4), current_lr)
            print(_statement)

        del images, labels, weights
        torch.cuda.empty_cache()

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch

def evaluate_model(model, val_loader, device, epoch, num_epochs, writer, log_file, current_lr, log_every=20):
    print('[EVALUATE] Evaluate model')
    _ = model.eval()

    y_trues = []
    y_preds = []
    losses = []

    with torch.no_grad():
        for i, (indices, images, labels, weights) in enumerate(val_loader):
            
            batch_loss = 0
            for j in range(len(images)):
                image = images[j].to(device)
                label = labels[j].to(device)
                weight = weights[j].to(device)

                prediction = model.forward(image.float())[0]

                loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
                batch_loss += loss

                probability = torch.sigmoid(prediction).item()
                y_trues.append(int(label.item()))
                y_preds.append(probability)
                
            loss_value = (batch_loss / len(images)).item()
            losses.append(loss_value)

            try:
                auc = metrics.roc_auc_score(y_trues, y_preds)
            except:
                auc = 0.5

            writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
            writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

            if i > 0 and i % log_every == 0:
                _statement = """[Epoch: {0} / {1} | Single batch number: {2} / {3}] avg_val_loss: {4} | val_auc: {5} | lr: {6}""".format(
                        epoch + 1, num_epochs, i, len(val_loader),
                        np.round(np.mean(losses), 4), np.round(auc, 4), current_lr
                    )
                print(_statement)

            del images, labels, weights
            torch.cuda.empty_cache()
    
    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)
    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    args = parse_arguments()
    run(args)