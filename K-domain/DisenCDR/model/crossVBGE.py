import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from model.GCN import VGAE
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


class crossVBGE(nn.Module):
    """
        GNN Module layer
    """

    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.opt = opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, UFEAs, UVs, VUs):
        for layer in self.encoder[:-1]:
            for i in range(len(UFEAs)):
                UFEAs[i] = F.dropout(UFEAs[i], self.dropout,
                                     training=self.training)
            UFEAs = layer(UFEAs, UVs, VUs)

        mean, sigma, = self.encoder[-1](UFEAs, UVs, VUs)
        return mean, sigma


class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt = opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"], nhid=opt["hidden_dim"], dropout=opt["dropout"], alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"], nhid=opt["hidden_dim"], dropout=opt["dropout"], alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], nhid=opt["feature_dim"], dropout=opt["dropout"], alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], nhid=opt["feature_dim"], dropout=opt["dropout"], alpha=opt["leakey"]
        )
        self.gcs_first = [GCN(nfeat=opt["feature_dim"], nhid=opt["hidden_dim"],
                              dropout=opt["dropout"], alpha=opt["leakey"]) for _ in range(self.opt['k'])]
        self.gcs_second = [GCN(nfeat=opt["hidden_dim"], nhid=opt["feature_dim"],
                               dropout=opt["dropout"], alpha=opt["leakey"]) for _ in range(self.opt['k'])]
        self.user_union = []
        for i in range(self.opt['k']):
            self.user_union.append(
                nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"]))

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        if self.opt["cuda"]:
            self.source_rate = self.source_rate.cuda()

    def forward(self, UFEAs, UVs, VUs):

        user_hos = [0 for i in range(self.opt['k'])]
        for i in range(self.opt['k']):
            user_hos[i] = self.gcs_first(UFEAs[i], VUs[i])
            user_hos[i] = self.gcs_second(user_hos[i], UVs[i])

        user_out = [0 for i in range(self.opt['k'])]
        for i in range(self.opt['k']):
            user_out[i] = torch.cat((user_hos[i], UFEAs[i]), dim=1)
            user_out[i] = self.user_union[i](source_User)
         
        val = 0
        for i in range(len(user_out)):
            for j in range(i+1, len(user_out)):
                val += self.source_rate * F.relu(user_out[i]) + (1-self.source_rate) * F.relu(user_out[j])
       
        return val

class LastLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt = opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gcs_first = [GCN(nfeat=opt["feature_dim"], nhid=opt["hidden_dim"],
                              dropout=opt["dropout"], alpha=opt["leakey"]) for _ in range(self.opt['k'])]
        self.gcs_second_mean = [GCN(nfeat=opt["hidden_dim"], nhid=opt["feature_dim"],
                               dropout=opt["dropout"], alpha=opt["leakey"]) for _ in range(self.opt['k'])]
        self.gcs_second_logstd = [GCN(nfeat=opt["hidden_dim"], nhid=opt["feature_dim"],
                               dropout=opt["dropout"], alpha=opt["leakey"]) for _ in range(self.opt['k'])]
                                
        self.user_union_mean = []
        self.user_union_logstd = []
        for i in range(self.opt['k']):
            self.user_union_mean.append(
                nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"]))
            self.user_union_logstd.append(
                nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"]))

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        if self.opt["cuda"]:
            self.source_rate = self.source_rate.cuda()

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
        gaussian_noise = torch.randn(
            mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(
            mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, UFEAs, UVs, VUs):
        user_hos_mean = [0 for i in range(self.opt['k'])]
        user_hos_logstd = [0 for i in range(self.opt['k'])]
    
        for i in range(self.opt['k']):
            user_hos_mean[i] = self.gcs_first(UFEAs[i], VUs[i])
            user_hos_mean[i] = self.gcs_second_mean(user_hos_mean[i], UVs[i])
        
            user_hos_logstd[i] = self.gcs_second_logstd(user_hos_logstd[i], UVs[i])
    
        user_means = [0 for i in range(self.opt['k'])]
        user_logstds = [0 for i in range(self.opt['k'])]
    
        for i in range(self.opt['k']):
            user_means[i] = torch.cat((user_hos_mean[i], UFEAs[i]), dim=1)
            user_means[i] = self.user_union_mean[i](user_means[i])
        
            user_logstds[i] = torch.cat((user_hos_logstd[i], UFEAs[i]), dim=1)
            user_logstds[i] = self.user_union_logstd[i](user_logstds[i])
    
        val_mean = 0
        val_logstd = 0
    
        for i in range(len(user_means)):
            for j in range(i+1, len(user_means)):
                val_mean += self.source_rate * user_means[i] + (1 - self.source_rate) * user_means[j]
                val_logstd += self.source_rate * user_logstds[i] + (1 - self.source_rate) * user_logstds[j]

        return val_mean, val_logstd
