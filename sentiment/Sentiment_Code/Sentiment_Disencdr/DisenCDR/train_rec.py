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
from model.trainer import CrossTrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper
import json
import codecs
import copy
import math

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--dataset', type=str, default='phone_electronic, sport_phone, sport_cloth, electronic_cloth', help='')

# model part
parser.add_argument('--model', type=str, default="DisenCDR", help="The model name.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--beta', type=float, default=0.9)
# train part
parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument("--sentiment", type=str, required=True, help="type of sentiment being used")

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
args.cuda = False
print(args.cuda)
if args.cpu:
    args.cuda = False
elif args.cuda:
    print("Yes")
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])


if "DisenCDR" in opt["model"]:
    folderName  = opt["dataset"]
    fileName = opt["sentiment"]
    source_graph = f"../dataset/{folderName}/" + "train_"+ fileName + ".txt"
    source_G = GraphMaker(opt, source_graph)
    source_UV = source_G.UV
    source_VU = source_G.VU
    source_adj = source_G.adj

    folderName = folderName.split("_")
    folderName = folderName[1] + "_" + folderName[0]
    target_train_data = f"../dataset/{folderName}/" + "train_"+ fileName + ".txt"
    target_G = GraphMaker(opt, target_train_data)
    target_UV = target_G.UV
    target_VU = target_G.VU
    target_adj = target_G.adj
    print("graph loaded!")


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)


print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
# for i, batch in enumerate(train_batch):
#     print(i, batch)
#     print(batch[0])
#     break
# exit()
source_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
# for i, batch in enumerate(source_dev_batch):
#     print("HERE")
#     print(i, batch)
#     print(batch[i])
#     exit()
# #     break

target_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)

print("user_num", opt["source_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("source train data : {}, target train data {}, source test data : {}, target test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data)))

if opt["cuda"]:
    source_UV = source_UV.cuda()
    source_VU = source_VU.cuda()
    source_adj = source_adj.cuda()

    target_UV = target_UV.cuda()
    target_VU = target_VU.cuda()
    target_adj = target_adj.cuda()

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

# exit()
# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct_graph(batch, source_UV, source_VU, target_UV, target_VU, source_adj, target_adj, epoch)
        train_loss += loss

    duration = time.time() - start_time
    train_loss = train_loss/len(train_batch)
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss, duration, current_lr))

    # if epoch % 10:
    #     # pass
    #     continue

    # eval model
    print("Evaluating on dev set...")
    trainer.model.eval()

    trainer.evaluate_embedding(source_UV, source_VU, target_UV, target_VU, source_adj, target_adj)
    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    rel_docs = 10 # number of releveant docs
    print("Source Dev Batches: ", len(source_dev_batch))
    for i, batch in enumerate(source_dev_batch):
        # print(i, batch)
        
        predictions, sentiment_scores = trainer.source_predict(batch)
        print("length of batch: ",len(batch[0]))
        # print(sentiment_scores)
        # print(predictions, sentiment_scores)
        # exit()
        scores_index = 0
        for pred in predictions:
            DCG = 0
            IDCG = 0
            valid_entity += 1
            
            # print(sentiment_scores[scores_index])
            sents = sentiment_scores[scores_index].tolist()
            # print("Sents", sents)
            rankings_list = (-pred).argsort()
            rank = (-pred).argsort().argsort()[0].item()
            # print("rank", rank)
            if(rank < rel_docs):
                HT+=1
                for j in range(len(rankings_list)):
                    
                    if(j == rel_docs):
                        break
                    

                    sent_score = sents[rankings_list[j]]
                    # print(sent_score)  
                    # print("Ranking list", (rankings_list[j],sent_score))                
                    DCG += sent_score / np.log2(j+2)
            # print("DCG: ", DCG)
            
            sents_sorted = sorted(sents, reverse=True)
            # print(sents_sorted)
            # exit()
            for j in range(rel_docs):
                IDCG += sents_sorted[j] / np.log2(j + 2)
            
            # print("IDCG: ", IDCG)
            
            if(IDCG != 0):
                NDCG += DCG / (IDCG)
                
            scores_index+=1
            
    print("Source samples: ", valid_entity)
    s_ndcg = NDCG / valid_entity
    print("Source Domain NDCG: ", s_ndcg)
    s_hit = HT / valid_entity
    print("Source domain HITS: ", s_hit)
    # exit()

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(target_dev_batch):
        
        predictions, sentiment_scores = trainer.target_predict(batch)
        # print(predictions, sentiment_scores)
        # exit()
        scores_index = 0
        for pred in predictions:
            DCG = 0
            IDCG = 0
            valid_entity += 1
            # print(sentiment_scores[scores_index])
            sents = sentiment_scores[scores_index].tolist()
            rankings_list = (-pred).argsort()
            # print(rankings_list)
            rank = (-pred).argsort().argsort()[0].item()
            if(rank < rel_docs):
                HT+=1
                # print("Hit", HT)
                for j in range(len(rankings_list)):
                    
                    if(j == rel_docs):
                        break
                    
                    sent_score = sents[rankings_list[j]]
                    # print("Sent Score", sent_score)
                    # print("Log term", np.log2(rankings_list[j]+2))
                    DCG += sent_score / np.log2(j+2)
                    # print("DCG", DCG)
            # print("DCG: ", DCG)
            
            sents_sorted = sorted(sents, reverse=True)
            # print(sents_sorted)
            # exit()
            for j in range(rel_docs):
                IDCG += sents_sorted[j] / math.log2(j + 2)
            
            # print("IDCG: ", IDCG)
            if(IDCG != 0):
                NDCG += DCG / (IDCG)

            scores_index+=1
    
    print("Target samples: ", valid_entity)
    t_ndcg = NDCG / valid_entity
    t_hit = HT / valid_entity
    print("Target Domain NDCG: ", t_ndcg)
    print("Target domain HITS: ", t_hit)

    print(
        "epoch {}: train_loss = {:.6f}, source_hit = {:.4f}, source_ndcg = {:.4f}, target_hit = {:.4f}, target_ndcg = {:.4f}".format(
            epoch, \
            train_loss, s_hit, s_ndcg, t_hit, t_ndcg))
    dev_score = t_ndcg
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
