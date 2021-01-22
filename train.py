import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from dataset.dataset import Data as data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.backends.cudnn as cudnn

from models import *
from utils.eval_utils import compute_accuracy
from utils.logger_utils import Logger
from config import options

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss = 0
        targets, outputs = [], []

        for batch_id, (data, target) in enumerate(train_loader):
            global_step += 1

            data = data.cuda()
            target = target.cuda()

            # Forward pass
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            train_loss += batch_loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (batch_id + 1) % options.disp_freq == 0:
                train_loss /= options.disp_freq
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))
                log_string("epoch: {0}, step: {1}, train_loss: {2:.4f} train_accuracy: {3:.02%}"
                           .format(epoch+1, batch_id+1, train_loss, train_acc))
                info = {'loss': train_loss,
                        'accuracy': train_acc}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = 0
                targets, outputs = [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss,
                                               best_acc=best_acc,
                                               global_step=global_step)
                net.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    net.eval()
    test_loss = 0
    targets, outputs = [], []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
        test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))

        # check for improvement
        loss_str, acc_str = '', ''
        if test_loss <= best_loss:
            loss_str, best_loss = '(improved)', test_loss
        if test_acc >= best_acc:
            acc_str, best_acc = '(improved)', test_acc

        # display
        log_string("validation_loss: {0:.4f} {1}, validation_accuracy: {2:.02%}{3}"
                   .format(test_loss, loss_str, test_acc, acc_str))

        # write to TensorBoard
        info = {'loss': test_loss,
                'accuracy': test_acc}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        # save checkpoint model
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'loss': test_loss,
            'acc': test_acc,
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
        log_string('--' * 30)
        return best_loss, best_acc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # bkp of model def
    os.system('cp {}/models/resnet.py {}'.format(BASE_DIR, save_dir))
    # bkp of train procedure
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/config.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    if options.model == 'resnet':
        net = resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
    elif options.model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
    elif options.model == 'densenet':
        net = densenet.densenet121(pretrained=True, drop_rate=0.2)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)
    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    os.system('cp {}/dataset/dataset.py {}'.format(BASE_DIR, save_dir))

    train_dataset = data(mode='train', data_len=options.data_len)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test', data_len=options.data_len)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
