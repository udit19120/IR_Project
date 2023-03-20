import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.singleVBGE import singleVBGE
from model.crossVBGE import crossVBGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


class DisenCDR(nn.Module):
    def __init__(self, opt):
        super(DisenCDR, self).__init__()
        self.opt = opt
        self.domain_specific_GNNs = []
        self.domain_share_GNNs = []

        for i in range(opt['k']):
            self.domain_specific_GNNs.append(singleVBGE(opt)) 
            self.domain_share_GNNs.append(singleVBGE(opt)) 

        self.share_GNN = crossVBGE(opt)  # gives q(ZuS|X,Y)
        self.dropout = opt["dropout"]

        print('making embeddings')

        self.domain_user_embeddings = []
        self.domain_item_embeddings = []
        self.domain_user_embeddings_share = []
        
        for i in range(opt['k']):
            self.domain_user_embeddings.append(nn.Embedding(opt[f"fname{i}_user_num"], opt["feature_dim"]))
            self.domain_item_embeddings.append(nn.Embedding(opt[f"fname{i}_item_num"], opt["feature_dim"]))
            self.domain_user_embeddings_share.append(nn.Embedding(opt[f"fname{i}_user_num"], opt["feature_dim"]))
        
        self.share_mean = nn.Linear(
            opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(
            opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.user_index = torch.arange(0, self.opt["fname0_user_num"], 1)
       
        self.domain_user_indices = []
        self.domain_item_indices = []
        
        for i in range(opt['k']):
            u_index = torch.arange(0, self.opt[f"fname{i}_user_num"], 1)
            i_index = torch.arange(0, self.opt[f"fname{i}_item_num"], 1)
            
            if self.opt["cuda"]:    #not changed cuz we're poor and we ain't having cuda
                u_index = u_index.cuda()
                i_index = i_index.cuda()
                
            self.domain_user_indices.append(u_index)
            self.domain_item_indices.append(i_index)

    # the below 2 methods deserve to be deleted as the third one is more generalized

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out
    
    def domain_predict_nns(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.predict_1(fea)
        out = F.relu(out)
        out = self.predict_2(out)
        out = torch.sigmoid(out)
        return out
    
    # the below 2 methods deserve to be deleted as the third one is more generalized

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def domain_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
    
    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        # gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"])
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(
            mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss

    def forward(self, UVs, VUs):
        
        domain_users = []
        domain_items = []
        domain_user_shares = []
        domain_learn_specific_users = []
        domain_learn_specific_items = []
        domain_user_means = []
        domain_user_sigmas = []
        
        for i in range(self.opt['k']):
            
            domain_users.append(self.domain_user_embeddings[i](self.domain_user_indices[i]))
            domain_items.append(self.domain_item_embeddings[i](self.domain_item_indices[i]))
            domain_user_shares.append(self.domain_user_embeddings_share[i](self.domain_user_indices[i]))
            # print(domain_user_shares[i].shape)
            a, b = self.domain_specific_GNNs[i](domain_users[i], domain_items[i], UVs[i], VUs[i])
            domain_learn_specific_users.append(a)
            domain_learn_specific_items.append(b)
            # domain_learn_specific_user, domain_learn_specific_item = domain_specific_GNN[i](domain_users[i], domain_items[i], UVs[i], VUs[i])
            a, b = self.domain_share_GNNs[i].forward_user_share(domain_users[i], UVs[i], VUs[i])
            domain_user_means.append(a)
            domain_user_sigmas.append(b)
             
        # source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(
        #     source_user, source_UV, source_VU)
        # target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(
        #     target_user, target_UV, target_VU)

        mean, sigma, = self.share_GNN(domain_user_shares, UVs, VUs)

        user_share, share_kld_loss = self.reparameters(mean, sigma)
        
        self.kld_loss = share_kld_loss
        for i in range(self.opt['k']):
            domain_share_kld = self._kld_gauss(mean, sigma, domain_user_means[i], domain_user_sigmas[i])
            self.kld_loss += self.opt['beta']*domain_share_kld
        
        # self.kld_loss = share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
        #     "beta"] * target_share_kld

        # source_learn_user = self.source_merge(torch.cat((user_share, source_learn_specific_user), dim = -1))
        # target_learn_user = self.target_merge(torch.cat((user_share, target_learn_specific_user), dim = -1))
        
        domain_learn_users = []
        for i in range(self.opt['k']):
            domain_learn_users.append(user_share + domain_learn_specific_users[i])
        # source_learn_user = 
        # target_learn_user = user_share + target_learn_specific_user

        return domain_learn_users, domain_learn_specific_items

    def warmup(self, UVs, VUs):
        
        domain_users = []
        domain_items = []

        for i in range(self.opt['k']):
            
            domain_users.append(self.domain_user_embeddings[i](self.domain_user_indices[i]))
            domain_items.append(self.domain_item_embeddings[i](self.domain_item_indices[i]))

        learn_specific_users = []
        learn_specific_items = []
        
        for i in range(self.opt['k']):
            a,b = self.domain_specific_GNNs[i](domain_users[i], domain_items[i], UVs[i], VUs[i])
            learn_specific_users.append(a)
            learn_specific_items.append(b)
        
        self.kld_loss = 0
        
        return learn_specific_users, learn_specific_items
