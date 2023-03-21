import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.trainercopy import CrossTrainer
from mutils.loader import DataLoader
from mutils.GraphMaker import GraphMaker
from mutils import torch_utils, helper
import json
import codecs
import copy


parser = argparse.ArgumentParser()
# dataset part
# parser.add_argument('--dataset', type=str, nargs = '+', default='cloth phone sport', help='List of arguments')
# parser.add_argument('--dataset', type=str, default='phone_electronic, sport_phone, sport_cloth, electronic_cloth, ', help='')
parser.add_argument('--k', type=int, default=2, help='')
# model part
parser.add_argument('--model', type=str, default="DisenCDR",
                    help="The model name.")
parser.add_argument('--feature_dim', type=int, default=128,
                    help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

parser.add_argument('--dropout', type=float, default=0.3,
                    help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9,
                    help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
# parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--beta', type=float, default=0.9)
# train part
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200,
                    help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt',
                    help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100,
                    help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str,
                    default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00',
                    help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true',
                    default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str,
                    help='Filename of the pretrained model.')


def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])


if "DisenCDR" in opt["model"]:
    G = []
    UV = []
    VU = []
    adj = []
    fnames = ['sport', 'cloth', 'phone']

    for i in range(opt['k']):
        filename = fnames[i]
        source_graph = "../dummy-dataset/" + filename + "/train.txt"
        g = GraphMaker(opt, source_graph)
        G += [g]
        UV += [g.UV]
        VU += [g.VU]
        adj += [g.adj]

    print("graph fully loaded!")


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

print("Loading data from {} with batch size {}...".format(
    fnames, opt['batch_size']))  # Generalised till here - Jahnvi
train_batch = DataLoader(fnames, opt['batch_size'], opt, evaluation=-1)
print('Data loaded of train batch')
dev_batches = []
for i in range(opt['k']):
    dev_batches.append(DataLoader(
        fnames, opt["batch_size"], opt, evaluation=i))
    print('Data loaded of dev batch for domain', i)

# source_dev_batch = DataLoader(fnames, opt["batch_size"], opt, evaluation = 1)
# print('Data loaded of source dev batch')
# target_dev_batch = DataLoader(fnames, opt["batch_size"], opt, evaluation = 2)
# print('Data loaded of target dev batch')


print("user_num", opt["fname0_user_num"])
for i in range(opt['k']):
    print(f"item_num_domain{i}", opt[f"fname{i}_item_num"])
    print("train data file{} : {}, test data file{} : {}".format(i, len(
        train_batch.train_datas[i]), i, len(train_batch.test_datas[i])))
# print("target_item_num", opt["target_item_num"])

if opt["cuda"]:
    for i in range(opt['k']):
        UV[i] = UV[i].cuda()
        VU[i] = VU[i].cuda()
        adj[i] = adj[i].cuda()

# model
if not opt['load']:
    trainer = CrossTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = CrossTrainer(opt)
    trainer.load(model_file)

dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']


# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct_graph(
            batch, UV, VU, adj, epoch)
        train_loss += loss

    duration = time.time() - start_time
    train_loss = train_loss/len(train_batch)
    print(format_str.format(datetime.now(), global_step, max_steps, epoch,
                            opt['num_epoch'], train_loss, duration, current_lr))

    # if epoch % 10:
    #     # pass
    #     continue

    # eval model
    print("Evaluating on dev set...")
    # ChatGPT: model.eval() disables dropout and batch normalization layer during inference ensuring model's behavior is consistent during training and inference.
    trainer.model.eval()

    trainer.evaluate_embedding(
        UV, VU, adj)

    ndcgs = []
    hits = []
    for i in range(opt['k']):
        NDCG = 0.0
        HT = 0.0
        valid_entity = 0.0
        for j, batch in enumerate(dev_batches[i]):
            predictions = trainer.source_predict(batch, i)
            for pred in predictions:
                rank = (-pred).argsort().argsort()[0].item()

                valid_entity += 1
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
                if valid_entity % 100 == 0:
                    print('.', end='')

        ndcgs.append(NDCG / valid_entity)
        hits.append(HT / valid_entity)

    print(
        "epoch {}: train_loss = {:.6f}".format(
            epoch,
            train_loss))
    for i in range(opt['k']):
        print("domain {} NDCG: {} HIT: {}".format(i, ndcgs[i], hits[i]))
    dev_score = ndcgs[0]
    print(max([dev_score] + dev_score_history))
    file_logger.log(
        "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history)))

    # save
    if epoch == 1 or dev_score > max(dev_score_history):
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        pass

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")
