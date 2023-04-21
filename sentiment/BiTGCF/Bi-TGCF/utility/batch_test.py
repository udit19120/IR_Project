'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time
# from utility.helper import ProgressBar
from tqdm import tqdm
from collections import defaultdict
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_sess = None
_data_type = None
_layer_size = None
_test_user_list = None


def test(sess, model, data_generator, test_user_list, data_type, batch_size, ks, layer_size):
    global _model
    global _sess
    global _testRatings
    global _testNegatives
    global _layer_size
    global _data_type
    global _test_user_list
    global _test_user_set
    global _test_item_set
    _test_user_set = data_generator.exist_users
    _test_item_set = data_generator.exist_items
    _model = model
    _sess = sess
    _testRatings = np.array(data_generator.ratingList)
    _testNegatives = data_generator.negativeList
    _layer_size = layer_size
    _data_type = data_type
    _test_user_list = test_user_list

    hits, ndcgs = [], []
    # need users_set, items_set
    users, items, user_gt_item = get_test_instance()
    print("Total samples in test: ", (len(users), len(items)))
    num_test_batches = len(users)//batch_size
    # bar_test = ProgressBar('test_'+_data_type, max=num_test_batches+1)
    # sample_id = 0

    test_preds = []
    if _data_type == 'source':
        for current_batch in tqdm(range(num_test_batches+1), desc='test_source', ascii=True):
            # bar_test.next()
            min_idx = current_batch*batch_size
            max_idx = np.min([(current_batch+1)*batch_size, len(users)])
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]
            print(type(batch_input_items))
            predictions = _sess.run(_model.scores_s, {_model.users_s: batch_input_users, _model.items_s: batch_input_items,
                                                      _model.node_dropout: [0.]*len(eval(_layer_size)),
                                                      _model.mess_dropout: [0.]*len(eval(_layer_size)),
                                                      _model.isTraining: True})

            test_preds.extend(predictions)
        assert len(test_preds) == len(users), 'source num is not equal'
    else:
        for current_batch in tqdm(range(num_test_batches+1), desc='test_target', ascii=True):
            # bar_test.next()
            min_idx = current_batch*batch_size
            max_idx = np.min([(current_batch+1)*batch_size, len(users)])
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]
            predictions = _sess.run(_model.scores_t, {_model.users_t: batch_input_users, _model.items_t: batch_input_items,
                                                      _model.node_dropout: [0.]*len(eval(_layer_size)),
                                                      _model.mess_dropout: [0.]*len(eval(_layer_size)),
                                                      _model.isTraining: True})

            test_preds.extend(predictions)
        assert len(test_preds) == len(users), 'target num is not equal'

    user_item_preds = defaultdict(lambda: defaultdict(float))
    user_pred_gtItem = defaultdict(float)
    for sample_id in range(len(users)):
        user = users[sample_id]
        item = items[sample_id]
        pred = test_preds[sample_id]  # [pos_prob, neg_prob]
        user_item_preds[user][item] = pred
    for user in user_item_preds.keys():
        item_pred = user_item_preds[user]
        hrs, nds = [], []
        for k in ks:
            # print("Item predictions: ",item_pred)
            sorted_item_pred = sorted(
                item_pred.items(), key=lambda x: x[1], reverse=True)
            # print("Positive Item: ", user_gt_item[user])
            # print("Num of items predictions: ", len(item_pred))
            ranklist = heapq.nlargest(k, item_pred, key=item_pred.get)
            # print("Ranking List: ", ranklist)

            hr = getHitRatio(ranklist, user_gt_item[user][0])
            ndcg = getNDCG(
                ranklist, user_gt_item[user][0], user_gt_item[user][1])
            hrs.append(hr)
            nds.append(ndcg)
        hits.append(hrs)
        ndcgs.append(nds)
    print("Total users in test:", len(hits))
    hr, ndcg = np.array(hits).mean(axis=0), np.array(ndcgs).mean(axis=0)
    return (hr, ndcg)


def get_test_instance():
    users, items = [], []
    user_gt_item = {}
    print(len(_test_user_list))
    for idx in _test_user_list:
        rating = _testRatings[idx]
        items_neg = _testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        gt_sent_score = rating[2]
        if(u not in _test_user_set or gtItem not in _test_item_set):
            continue
        user_gt_item[u] = [gtItem, gt_sent_score]
        for item in items_neg:
            users.append(u)
            items.append(item)
        items.append(gtItem)
        users.append(u)
    return np.array(users), np.array(items), user_gt_item


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem, gt_sentiment_score):
    idcg = 0
    dcg = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        s_score = 0
        if(item == gtItem):
            s_score = gt_sentiment_score
        dcg += s_score/(math.log2(i+2))

    sents_sorted = [0]*len(ranklist)
    sents_sorted[0] = gt_sentiment_score
    for j in range(len(sents_sorted)):
        idcg += sents_sorted[j] / np.log2(j + 2)

    ndcg = dcg/idcg

    return ndcg
