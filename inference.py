import os
import warnings

import h5py

warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.dataset import Data as data
from utils.visualize_utils import visualize
from utils.eval_utils import compute_accuracy
from models import *
from config import options
import pandas as pd
import torch.nn.functional as F
from utils.eval_utils import mutual_info, predictive_entropy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# def apply_dropout(m):
#     if m.__class__.__name__ == '_DenseLayer':
#         m.train()


def evaluate():
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
        targets_temp = torch.cat(targets).cpu().numpy()
        outputs_temp = np.argmax(torch.cat(outputs).cpu().numpy(), axis=1)
        log_string('Glomerulus Level Classification Confusion Matrix and Accuracy: ')
        log_string(str(confusion_matrix(targets_temp, outputs_temp)))

        # display
        log_string("validation_loss: {0:.4f}, validation_accuracy: {1:.02%}"
                   .format(test_loss, test_acc))
    return outputs, targets, test_loss

@torch.no_grad()
def mc_evaluate():
    net.eval()

    # if options.MC:
    #     net.apply(apply_dropout)

    test_loss = 0
    targets, outputs, probs = [], [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            prob = F.softmax(output)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            probs += [prob]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
    return torch.cat(probs).unsqueeze(0).cpu().numpy(), F.one_hot(torch.cat(targets), options.num_classes).cpu().numpy(), test_loss

if __name__ == '__main__':

    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))
    mc_dir = os.path.join(save_dir, 'mc_results')
    if not os.path.exists(mc_dir):
        os.makedirs(mc_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    if options.model == 'resnet':
        net = resnet.resnet50()
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
        grad_cam_hooks = {'forward': net.layer4, 'backward': net.fc}
    elif options.model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
        grad_cam_hooks = {'forward': net.features, 'backward': net.fc}
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
        grad_cam_hooks = {'forward': net.Mixed_7c, 'backward': net.fc}
    elif options.model == 'densenet':
        DROP = 0.1
        net = densenet.densenet121(pretrained=True, drop_rate=DROP)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)
        grad_cam_hooks = {'forward': net.features.norm5, 'backward': net.classifier}

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

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

    # train_dataset = data(mode='train', data_len=options.data_len)
    # train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
    #                           shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test', data_len=options.data_len)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start Testing')

    test = np.load('<your_test_np_file_dir>'
                   .format(options.loos))
    # x = np.transpose(test['x'][:], [0, 3, 1, 2]).astype(int)
    labels = test['y'][:].astype(int)
    names = test['name']
    subjects = test['subject']
    file_name = test['file_name']
    file_name = ['<your_dataset_np_file_dir>' + s for s in file_name]
    file_names = file_name

    if options.MC:
        mc_probs = []
        temp=[]
        for mc_iter in range(options.mc_iter):
            print('running mc iteration # {}'.format(mc_iter+1))
            iter_probs, iter_targets, iter_loss = mc_evaluate()
            mc_probs += [iter_probs]
            # temp = np.concatenate(mc_probs)
            # mean_prob = temp.mean(axis=0)
            # var_pred_entropy = predictive_entropy(mean_prob)
            # var_pred_MI = mutual_info(temp)
            # acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), iter_targets.argmax(axis=1))) / \
            #       mean_prob.shape[0]
            # if options.GAN == 0:
            #     ST_SET = 0
            # else:
            #     ST_SET = options.num_gan
            # np.savez_compressed(os.path.join(options.mc_dir,
            # 'output_{}_loos={}_mciter={}_GAN={}'.format(DROP, options.loos,
            #     mc_iter+1, ST_SET)), y=iter_targets, data=mc_probs)
            # print('Filename: ' + 'output_{}_loos={}_mciter={}_GAN={}.npz'.format(DROP, options.loos,
            #     mc_iter+1, ST_SET))
            # print('accuracy={0:.05%}'.format(acc))
        mc_probs = np.concatenate(mc_probs)  # [mc_iter, N, C]

        if options.GAN == 0:
            ST_SET = 0
        else:
            ST_SET = options.num_gan

        mean_prob = mc_probs.mean(axis=0)
        var_pred_entropy = predictive_entropy(mean_prob)
        var_pred_MI = mutual_info(mc_probs)
        np.savez_compressed(os.path.join(options.mc_dir, 'output_{}_loos={}_mc={}_GAN={}'.format(DROP,options.loos, options.mc_iter, ST_SET)),
                            y=iter_targets, data=mc_probs, names=names, subjects=subjects, file_names=file_names, unc=var_pred_entropy)

        print('Filename: ' + 'output_{}_loos={}_mciter={}_GAN={}.npz'.format(DROP, options.loos,
                                                                                  options.mc_iter, ST_SET))


        acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), iter_targets.argmax(axis=1))) / mean_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))
        exit()
    else:
        test_outputs, test_targets, test_loss_ = evaluate()
        test_acc = compute_accuracy(torch.cat(test_targets), torch.cat(test_outputs))
        targets = torch.cat(test_targets).cpu().numpy()
        outputs = np.argmax(torch.cat(test_outputs).cpu().numpy(), axis=1)

        # h5f = h5py.File(os.path.join(save_dir, 'prediction_{}.h5'.format(options.loos)), 'w')
        # h5f.create_dataset('x', data=x)
        # h5f.create_dataset('name', data=names)
        # h5f.create_dataset('y', data=targets)
        # h5f.create_dataset('y_pred', data=outputs)
        # h5f.close()

    #################################
    # Patient Level classifiers
    #################################

    # test_pd = pd.DataFrame(list(zip(names, outputs, targets)), columns=['Names', 'Outputs', 'Targets'])
    # test_pd['True Targets'] = ""
    #
    # for j in range(len(test_pd['Names'])):
    #     if test_pd['Names'][j] in [192066, 1746372, 197738, 191664]:
    #         test_pd['True Targets'][j] = 0
    #     elif test_pd['Names'][j] in [1730606, 1734349, 1738311, 199917, 1821954, 1735395, 1731598]:
    #         test_pd['True Targets'][j] = 1
    #
    #
    # # Generating Patient label dataframe
    # patients_ids = test_pd['Names'].unique()
    # classifier_set = pd.DataFrame(columns=['Output Label 0', 'Output Label 1',
    #                                        'Label Targets', 'True Targets', 'Label Prediction'], index=patients_ids)
    # for i in range(len(patients_ids)):
    #     classifier_set.at[patients_ids[i], 'Label Targets'] = test_pd[test_pd['Names'] == patients_ids[i]]['Targets'].mode().values.squeeze()
    #     classifier_set.at[patients_ids[i], 'True Targets'] = test_pd[test_pd['Names'] == patients_ids[i]]['True Targets'].mode().values.squeeze()
    #     for j in range(options.num_classes):
    #         try:
    #             classifier_set.at[patients_ids[i], 'Output Label {}'.format(j)] = test_pd[test_pd['Names'] == patients_ids[i]]['Outputs'].value_counts(normalize=True)[j].astype(float)
    #         except KeyError:
    #             classifier_set.at[patients_ids[i], 'Output Label {}'.format(j)] = 0
    #
    #     # Generate Label Prediction
    #     if options.num_classes == 2:
    #         labels = ['Output Label 0', 'Output Label 1']
    #         if classifier_set.iloc[i]['Output Label 0'] == classifier_set[labels].iloc[i].max():
    #             classifier_set.at[patients_ids[i], 'Label Prediction'] = 0
    #         elif classifier_set.iloc[i]['Output Label 1'] == classifier_set[labels].iloc[i].max():
    #             classifier_set.at[patients_ids[i], 'Label Prediction'] = 1
    #
    # # Label Prediction analysis
    # log_string('----------------------')
    # log_string('Patient Label Classification confusion Matrix:')
    # log_string(str(confusion_matrix(classifier_set['Label Targets'].astype(int).values,
    #                                 classifier_set['Label Prediction'].astype(int).values)))
    # log_string('Patient Label Classification Accuracy: ' + str((accuracy_score(classifier_set['Label Targets'].astype(int).values,
    #                                 classifier_set['Label Prediction'].astype(int).values))*100) + '%')
    # log_string('----------------------')
    #
    # # Patient by Patient analysis
    # for i in range(len(patients_ids)):
    #     log_string('Predicted Label Counts for Patient ID: ' + str(patients_ids[i]))
    #     for j in range(options.num_classes):
    #         log_string('Label ' + str(j) + ': ' + str(sum(test_pd[test_pd['Names'] == patients_ids[i]]['Outputs'] == j)))
    #     log_string('Patient Label: ' + str(classifier_set.iloc[i]['Label Targets']))
    #     log_string('Patient Labeled Correctly using Majority Classification: ' + str(classifier_set.iloc[i]['Label Targets'] == classifier_set.iloc[i]['Label Prediction']))
    #     log_string('----------------------')

    #################################
    # Grad Cam visualizer
    #################################
    if options.gradcam:
        log_string('Generating Gradcam visualizations')
        iter_num = options.load_model_path.split('/')[-1].split('.')[0]
        img_dir = os.path.join(save_dir, 'imgs')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        viz_dir = os.path.join(img_dir, iter_num)
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        visualize(net, test_loader, grad_cam_hooks, viz_dir)
        log_string('Images saved in: {}'.format(viz_dir))



