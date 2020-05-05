import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self._gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def _gram_matrix(self, input):
        # a=batch size (=1)
        # b=number of feature maps
        # (c,d)=dimension of a f. map (N=c*d)
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())
        # we normalize the values of the gram matrix
        # by diving by the number of element in each feature maps.
        return G.div(a * b * c * d)
