"""
This script implements an outlier interpretation method of the following paper:
"Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network". in WWW'21.
@ Author: Hongzuo Xu
@ email: hongzuo.xu@gmail.com or leogarcia@126.com or xuhongzuo13@nudt.edu.cn
"""

from pyod.models import lscp
import numpy as np
import time, math
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from tqdm import tqdm
from model_qml.utils import EarlyStopping, min_max_normalize

from model_qml.datasets import MyNewSingleQuitupletSelector
from model_qml.datasets import NewSingleQuintupletDataset
from model_qml.networks import PMLnet
from model_qml.networks import MyLoss


class PML:
    def __init__(self, nbrs_num=10, rand_num=10, anom_num=10, 
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=2, margin_add = 1,
                 verbose=True, gpu=True):
        self.verbose = verbose

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        # a list of normal nbr of each anomaly
        self.normal_nbr_indices = []
        self.anom_nbr_indices = [] #here I init a [] to store the k-nearest neighbors anomalies of ano_idx

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda and gpu else "cpu")
        if cuda:
            torch.cuda.set_device(0)
        print("device:", self.device)

        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.anom_num = anom_num

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_linear = n_linear
        self.margin = margin
        self.margin_add = margin_add
        return

    def fit(self, x, y):
        device = self.device

        self.dim = x.shape[1]
        x = min_max_normalize(x)
        self.ano_idx = np.where(y == 1)[0]

        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)
        self.prepare_nbrs()
        self.prepare_abnomal_nbrs()

        # train model for each anomaly
        attn_lst, W_lst = [], []
        if self.verbose:
            iterator = range(len(self.ano_idx))
        else:
            iterator = tqdm(range(len(self.ano_idx)))
        for ii in iterator:
            idx = self.ano_idx[ii]

            s_t = time.time()
            relation, W = self.interpret_ano_with_quintuplet(ii) #here is the enterance of the code
            attn_lst.append(relation)
            W_lst.append(W)

            if self.verbose:
                print("Ano_id:[{}], ({}/{}) \t time: {:.2f}s\n".format(
                    idx, (ii + 1), len(self.ano_idx),
                    (time.time() - s_t)))

        fea_weight_lst = [] 
        for ii, idx in enumerate(self.ano_idx):
            relation, w = attn_lst[ii], W_lst[ii]
            fea_weight = np.zeros(self.dim) #特征权重

            # attention (linear space) + w --> feature weight (original space)
            for j in range(len(relation)):
                fea_weight += relation[j] * abs(w[j])
            fea_weight_lst.append(fea_weight) #compute the weight of each anomaly's feature 
        return fea_weight_lst


    #this is the first test 
    def interpret_ano_with_quintuplet(self,ii):
        idx = self.ano_idx[ii]
        device = self.device
        dim = self.dim
        normal_nbr_indices = self.normal_nbr_indices[ii]#each outlier has a set of k-nearst neighbors which are abnomal
        anom_nbr_indices = self.anom_nbr_indices[ii] #each anom has a set of k-nearest neighbors which are abnomal
        data_loader, test_loader = self.prepare_quintuplets(idx, nbr_indices=normal_nbr_indices, anom_nbr_indices=anom_nbr_indices) #get the triplets
        n_linear = self.n_linear #d
        model = PMLnet(n_feature=dim, n_linear=n_linear)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = MyLoss(device, margin1=self.margin, margin2=self.margin_add)

        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        early_stp = EarlyStopping(patience=3, verbose=False)

        for epoch in range(self.n_epoch):
            model.train()
            total_loss = 0
            es_time = time.time()

            batch_cnt = 0
            for anchor, pos, neg, neg2, pos2 in data_loader: #iterate triplets and 
                anchor, pos, neg, neg2, pos2 = anchor.to(device), pos.to(device), neg.to(device), neg2.to(device), pos2.to(device)
                anchor, positive, negative, negative2, positive2, relation =  model(anchor, pos, neg, neg2, pos2)
                loss = criterion(anchor, positive, negative, negative2, positive2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_cnt += 1

            train_loss = total_loss/ batch_cnt
            est = time.time() - es_time

            if self.verbose and (epoch + 1) % 1 == 0:
                message = 'Epoch: [{:02}/{:02}]  loss: {:.4f} Time: {:.2f}s'.format(epoch + 1, self.n_epoch,
                                                                                    train_loss, est)
                print(message)
            scheduler.step()

            early_stp(train_loss, model)
            if early_stp.early_stop:
                model.load_state_dict(torch.load(early_stp.path))
                if self.verbose:
                    print("early stopping")
                break

        # distill W and relation from network
        for anchor, pos, neg, neg2, pos2 in test_loader:
            model.eval()
            anchor, pos, neg, neg2, pos2 = anchor.to(device), pos.to(device), neg.to(device), neg2.to(device), pos2.to(device)
            _, _, _, _, _, relation= model(anchor, pos, neg, neg2, pos2)

        attn_avg = torch.mean(relation, dim=0)
        attn_avg = attn_avg.data.cpu().numpy()
        W = model.linear.weight.data.cpu().numpy()
        return attn_avg, W

    def prepare_quintuplets(self, idx, nbr_indices, anom_nbr_indices):
        x= self.x
        y = self.y
        selector = MyNewSingleQuitupletSelector(rand_num=self.rand_num, anom_num=self.anom_num, nbr_indices=nbr_indices, anom_nbr_indices=anom_nbr_indices)
        dataset = NewSingleQuintupletDataset(idx,x,y,quatruplet_selector=selector)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader
    
#generate k-neighbor normal instances of outlier ii
    def prepare_nbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        x_anom = x[anom_idx]
        noml_idx = np.where(y == 0)[0]
        x_noml = x[noml_idx]

        #judge the number of the whole anomalous instances
        if len(x_noml) <= self.nbrs_num:
            n_neighbors = max(len(x_noml) - 1, 1)
        else:
            n_neighbors = self.nbrs_num

        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(x_noml)
        tmp_indices = nbrs_local.kneighbors(x_anom)[1]

        for idx in tmp_indices:
            nbr_indices = noml_idx[idx]
            self.normal_nbr_indices.append(nbr_indices) #here append the k-neighbors
        return
#gennerate k-neighbor abnormal instances of outlier ii
    def prepare_abnomal_nbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        x_anom = x[anom_idx]

        #judge the number of the whole anomalous instances
        if len(x_anom) <= self.nbrs_num:
            n_neighbors = max(len(x_anom) - 1, 1)
        else:
            n_neighbors = self.anom_num
        

        nbrs_anomalous = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x_anom)
        tmp_indices = nbrs_anomalous.kneighbors(x_anom)[1]  #get the indices of the k-neighbors and omit the return distances

        for idx in tmp_indices:
            anomalous_nbr_indices = anom_idx[idx][1:] #remove the first indices, since the first one is the anomaly itself
            self.anom_nbr_indices.append(anomalous_nbr_indices)
        return

