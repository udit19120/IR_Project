"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, fnames, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation

        self.ma_sets = []
        self.ma_lists = []
        self.train_datas = []
        self.test_datas = []
        self.users = []
        self.items = []

        for i in range(opt['k']):

            train_data = "../Code/" + fnames[i] + "/train_sentiment.txt"
            test_data = "../Code/" + fnames[i] + "/test_sentiment.txt"

            a, b, c, d, e, f = self.read_data(train_data, test_data)

            # print("a", a)
            # print("b", b)
            # print("c", c)
            # print("d", d)
            # print("e", e)
            # print("f", f)

            self.ma_sets.append(a)
            self.ma_lists.append(b)
            self.train_datas.append(c)
            self.test_datas.append(d)
            self.users.append(e)
            self.items.append(f)

            opt[f"fname{i}_user_num"] = len(e)
            opt[f"fname{i}_item_num"] = len(f)

        opt["rate"] = self.rate()

        # assert opt["source_user_num"] == opt["target_user_num"]
        if evaluation == -1:
            data = self.preprocess()    # Error here
        else:
            data = self.preprocess_for_predict()
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            # Just to handle batch size till
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data) % batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
            # here
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_data(self, train_file, test_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = {}
            item = {}
            ma = {}
            ma_list = {}
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    user[line[0]] = len(user)
                if item.get(line[1], "zxczxc") is "zxczxc":
                    item[line[1]] = len(item)
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                train_data.append([line[0], line[1]])
                if line[0] not in ma:
                    ma[line[0]] = set()
                    ma_list[line[0]] = []
                ma[line[0]].add(line[1])
                ma_list[line[0]].append(line[1])
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    continue
                if item.get(line[1], "zxczxc") is "zxczxc":
                    continue
                line[0] = user[line[0]]
                line[1] = item[line[1]]

                ret = [line[1]]
                for i in range(99):
                    while True:
                        rand = random.randint(0, len(item)-1)
                        if rand in ma[line[0]]:
                            continue
                        ret.append(rand)
                        break
                test_data.append([line[0], ret])

        return ma, ma_list, train_data, test_data, user, item

    def rate(self):
        # ret = []
        # for i in range(len(self.source_ma_set)):
        #     ret = len(self.source_ma_set[i]) / (len(self.source_ma_set[i]) + len(self.target_ma_set[i]))
        # return ret
        ret = []
        # print(ret.shape)
        tot_len_list = []
        for i in range(len(self.ma_sets[0])):
            total_len = 0
            for j in range(self.opt['k']):
                # print(total_len, self.ma_sets[j][i])
                total_len += len(self.ma_sets[j][i])

            tot_len_list.append(total_len)

        for i in range(self.opt['k']):
            v = np.zeros((len(self.ma_sets[0]), 1))
            for j in range(len(self.ma_sets[0])):
                v = len(self.ma_sets[i][j])/tot_len_list[j]
            ret.append(v)
        print("Im here in mutils")

        return ret

    def preprocess_for_predict(self):
        processed = []
        for d in self.test_datas[self.eval]:
            # user, item_list(pos in the first node)
            processed.append([d[0], d[1]])
        return processed
        # return [1,2]
        # pass

    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        # print(self.train_datas[0])

        # for d in self.source_train_data:
        #     d = [d[1], d[0]]
        #     processed.append(d + [-1])
        # for d in self.target_train_data:
        #     processed.append([-1] + d)
        for i in range(self.opt['k']):
            for d in self.train_datas[i]:
                processed.append([i]+d)
        return processed

    def find_pos(self, ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval != -1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else:
            # source_neg_tmp = []
            # target_neg_tmp = []
            # source_pos_tmp = []
            # target_pos_tmp = []
            neg_tmps = [[] for _ in range(self.opt['k'])]
            pos_tmps = [[] for _ in range(self.opt['k'])]
            user = []
            for b in batch:
                # if b[0] == -1:  # target train sample [-1,user,item]
                #     source_pos_tmp.append(
                #         self.find_pos(self.source_ma_list, b[1]))
                #     target_pos_tmp.append(b[2])
                # else:  # source train sample [item,user,-1]
                #     source_pos_tmp.append(b[0])
                #     target_pos_tmp.append(
                #         self.find_pos(self.target_ma_list, b[1]))
                # source_neg_tmp.append(self.find_neg(
                #     self.source_ma_set, b[1], "source_item_num"))
                # target_neg_tmp.append(self.find_neg(
                #     self.target_ma_set, b[1], "target_item_num"))
                for i in range(self.opt['k']):
                    if(b[0] == i):
                        pos_tmps[i].append(b[2])
                    else:
                        pos_tmps[i].append(
                            self.find_pos(self.ma_lists[i], b[1]))
                    neg_tmps[i].append(self.find_neg(
                        self.ma_sets[i], b[1], f"fname{i}_item_num"))
                user.append(b[1])
            # return (torch.LongTensor(user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp))
            return (torch.LongTensor(user), [torch.LongTensor(pos_temp) for pos_temp in pos_tmps], [torch.LongTensor(neg_temp) for neg_temp in neg_tmps])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
