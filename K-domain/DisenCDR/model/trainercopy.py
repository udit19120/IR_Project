import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.DisenCDRJcopy import DisenCDR
from mutils import torch_utils


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "DisenCDR":
            print('making disencdr')
            self.model = DisenCDR(opt)
        else:
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(
            opt['optim'], [{"params": self.model.parameters()}]+self.model.params_dicts, opt['lr'])
        self.epoch_rec_loss = []

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            # inputs = [Variable(b.cuda()) for b in batch]
            # user = inputs[0]
            # pos_item = inputs[1]
            # neg_item = inputs[2]
            inputs = batch
            user = Variable(inputs[0].cuda())
            pos_items = [Variable(p.cuda()) for p in inputs[1]]
            neg_items = [Variable(n.cuda()) for n in inputs[2]]
        else:
            #     inputs = [Variable(b) for b in batch]
            #     user = inputs[0]
            #     pos_item = inputs[1]
            #     neg_item = inputs[2]
            # return user, pos_item, neg_item
            inputs = batch
            # inputs = [Variable(b) for b in batch]
            user = Variable(inputs[0])
            pos_items = [Variable(p) for p in inputs[1]]
            neg_items = [Variable(n) for n in inputs[2]]
        return user, pos_items, neg_items

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def source_predict(self, batch, i):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.user[i], user_index)
        item_feature = self.my_index_select(self.item[i], item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.source_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.target_user, user_index)
        item_feature = self.my_index_select(self.target_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.target_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def evaluate_embedding(self, UV, VU, adj=None):
        self.user, self.item = self.model(UV, VU)

    def for_bcelogit(self, x):
        y = 1 - x
        return torch.cat((x, y), dim=-1)

    def calculate_elbo_loss(self, scores, labels):
        loss = 0
        for i in range(len(scores)):
            loss += self.criterion(scores[i], labels[i])

        return loss

    def calculate_KLD_loss(self):

        loss = 0
        for i in range(self.opt['k']):
            loss += self.model.domain_specific_GNNs[i].encoder[-1].kld_loss

        return loss

    def reconstruct_graph(self, batch, UV, VU, adj=None, epoch=100):
        self.model.train()
        # print(epoch)
        # print(self.model.params_dicts[0])
        self.optimizer.zero_grad()

        # user, source_pos_item, source_neg_item, target_pos_item, target_neg_item = self.unpack_batch(batch)
        user, pos_item, neg_item = self.unpack_batch(batch)

        if epoch < 2:
            # self.source_user, self.source_item, self.target_user, self.target_item = self.model.warmup(source_UV,source_VU,target_UV,target_VU)
            self.user, self.item = self.model.warmup(UV, VU)
        else:
            # self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV,source_VU, target_UV,target_VU)
            self.user, self.item = self.model(UV, VU)

        K_user_features = []
        K_item_pos_features = []
        K_item_neg_features = []
        for i in range(self.opt['k']):
            user_feature = self.my_index_select(self.user[i], user)
            item_pos_feature = self.my_index_select(self.item[i], pos_item[i])
            item_neg_feature = self.my_index_select(self.item[i], neg_item[i])

            K_user_features.append(user_feature)
            K_item_pos_features.append(item_pos_feature)
            K_item_neg_features.append(item_neg_feature)

        pos_scores = []
        neg_scores = []
        for i in range(self.opt['k']):
            pos_score = self.model.source_predict_dot(
                K_user_features[i], K_item_pos_features[i])
            neg_score = self.model.source_predict_dot(
                K_user_features[i], K_item_neg_features[i])

            pos_scores.append(pos_score)
            neg_scores.append(neg_score)

        pos_labels = []
        neg_labels = []
        for i in range(self.opt['k']):
            pos_labels.append(torch.ones(pos_scores[i].size()))
            neg_labels.append(torch.zeros(neg_scores[i].size()))

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        self.ELBO_loss = self.calculate_elbo_loss(
            pos_scores, pos_labels) + self.calculate_elbo_loss(neg_scores, neg_labels)
        self.K_KLD_loss = self.calculate_KLD_loss()  # change according to Jahnvi's code

        loss = self.ELBO_loss + self.K_KLD_loss + self.model.kld_loss
        # loss = self.K_KLD_loss

        loss.backward()
        self.optimizer.step()
        return loss.item()
