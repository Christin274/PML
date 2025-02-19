


import torch
import torch.nn as nn
import torch.nn.functional as F

class PMLnet(nn.Module):
    def __init__(self, n_feature, n_linear):
        super(PMLnet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)
        self.relation= torch.nn.Linear(5* n_linear, n_linear)
        self.layer_norm = torch.nn.LayerNorm(n_linear)
         # 学习每个关系的权重


    def forward(self, anchor, positive, negative, negative2, positive2):
        anchor = self.linear(anchor)
        positive = self.linear(positive)
        negative = self.linear(negative)
        negative2 = self.linear(negative2)
        positive2 = self.linear(positive2)

        dis_anchor_positive = torch.norm(anchor - positive,p=2,dim=1)
        dis_anchor_negative = torch.norm(anchor - negative,p=2,dim=1)
        relation_factor = torch.sigmoid(dis_anchor_negative-dis_anchor_positive)

        x = torch.cat([negative, anchor, positive, negative2, positive2], dim=1)

        x = torch.relu(x)
        x = self.relation(x)
        #improve
        relation = self.layer_norm(x)


        _min = torch.unsqueeze(torch.min(x, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(x, dim=1)[0], 0).t()
        relation = (relation - _min) / (_max - _min)

        relation= relation * relation_factor.unsqueeze(-1)
        

        embedded_n = negative * relation
        embedded_a = anchor * relation 
        embedded_p = positive * relation 
        embedded_n2 = negative2 * relation 
        embedded_p2 = positive2 * relation 

        return embedded_a, embedded_p, embedded_n, embedded_n2, embedded_p2, relation

    def get_lnr(self, x):
        return self.linear(x)





class MyLoss(nn.Module):
    """
    this is the loss of my first test
    """

    def __init__(self, device, margin1, margin2):
        super(MyLoss, self).__init__()
        self.device = device
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin1, p = 2)
        self.criterion_tml_add = torch.nn.TripletMarginLoss(margin=margin2, p = 2)
        self.sigma_weight = nn.Parameter(torch.Tensor(2, 1))
        nn.init.xavier_uniform_(self.sigma_weight)
        self.softplus = torch.nn.Softplus()

    def forward(self, embed_anchor, embed_pos, embed_neg, embed_neg2, embed_pos2):
      
        loss_tml1 = self.criterion_tml(embed_neg, embed_neg2, embed_anchor) 
        loss_tml2 = self.criterion_tml_add(embed_pos, embed_pos2, embed_anchor)

    
        sigma_weight_0 = self.softplus(self.sigma_weight[0].to(self.device))
        sigma_weight_1 = self.softplus(self.sigma_weight[1].to(self.device))


        loss_former = (1/(2*sigma_weight_0**2))*loss_tml1 + (1/(2*sigma_weight_1**2))*loss_tml2 
        regularization = torch.log(sigma_weight_0 * sigma_weight_1)
        loss = loss_former + regularization
        return loss



class MyLoss_abaltion1(nn.Module):
    """
    Optimized loss function combining triplet losses.
    """
    def __init__(self, device, margin1, margin2):
        super(MyLoss_abaltion1, self).__init__()
        self.device = device
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin1, p = 2)
        self.criterion_tml_add = torch.nn.TripletMarginLoss(margin=margin2, p = 2)
        self.sigma_weight = nn.Parameter(torch.Tensor(2, 1))
        nn.init.xavier_uniform_(self.sigma_weight)
        self.softplus = torch.nn.Softplus()

    def forward(self, embed_anchor, embed_pos, embed_neg, embed_neg2, embed_pos2):
      
        loss_tml1 = self.criterion_tml(embed_neg, embed_neg2, embed_anchor) 
        loss_tml2 = self.criterion_tml_add(embed_pos, embed_pos2, embed_anchor)

    
        sigma_weight_0 = self.softplus(self.sigma_weight[0].to(self.device))
        sigma_weight_1 = self.softplus(self.sigma_weight[1].to(self.device))


        loss_former = (1/(2*sigma_weight_0**2))*loss_tml1 + (1/(2*sigma_weight_1**2))*loss_tml2 
        regularization = torch.log(sigma_weight_0 * sigma_weight_1)
        loss = loss_former + regularization
        return loss


class MyLoss_abaltion2(nn.Module):
    """
    Optimized loss function combining triplet losses.
    """
    def __init__(self, device, margin1, margin2):
        super(MyLoss_abaltion2, self).__init__()
        self.device = device
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin1, p = 2)
        self.criterion_tml_add = torch.nn.TripletMarginLoss(margin=margin2, p = 2)

    def forward(self, embed_anchor, embed_pos, embed_neg, embed_neg2, embed_pos2):
      
        loss_tml1 = self.criterion_tml(embed_neg, embed_neg2, embed_anchor) 
        loss_tml2 = self.criterion_tml_add(embed_pos, embed_pos2, embed_anchor)

        loss = loss_tml1
        return loss
        


# ---------------------- QML - ablation1 -------------------------- #
# without quituplet feature representation module
class PMLabla1net(nn.Module):
    def __init__(self, n_feature, n_linear):
        super(PMLabla1net, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)

    def forward(self, anchor, positive, negative, negative2, positive2):
            anchor = self.linear(anchor)
            positive = self.linear(positive)
            negative = self.linear(negative)
            negative2 = self.linear(negative2)
            positive2 = self.linear(positive2)


            return anchor, positive, negative, negative2, positive2

    def get_lnr(self, x):
        return self.linear(x)


# -------------------------- QML - ablation2 ------------------------------ #
# test the significance of multi-task loss function

class PMLabla2net(nn.Module):
    def __init__(self, n_feature, n_linear):
        super(PMLabla2net, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)
        self.relation= torch.nn.Linear(5* n_linear, n_linear)
        self.layer_norm = torch.nn.LayerNorm(n_linear)
         # 学习每个关系的权重


    def forward(self, anchor, positive, negative, negative2, positive2):
        anchor = self.linear(anchor)
        positive = self.linear(positive)
        negative = self.linear(negative)
        negative2 = self.linear(negative2)
        positive2 = self.linear(positive2)

        x = torch.cat([negative, anchor, positive, negative2, positive2], dim=1)

        x = torch.relu(x)
        x = self.relation(x)
        #improve
        relation = self.layer_norm(x)


        _min = torch.unsqueeze(torch.min(x, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(x, dim=1)[0], 0).t()
        relation = (x - _min) / (_max - _min)
        

        embedded_n = negative * relation
        embedded_a = anchor * relation 
        embedded_p = positive * relation 
        embedded_n2 = negative2 * relation 
        embedded_p2 = positive2 * relation 

        return embedded_a, embedded_p, embedded_n, embedded_n2, embedded_p2, relation

    def get_lnr(self, x):
        return self.linear(x)
