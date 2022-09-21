# This code extends Facebooks' SimSiam code for paired data and
# is heavily based on it
# License of SimSiam and code can be found at the SimSiam repo: TBA

import torch
import torch.nn as nn


class PairedSimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(PairedSimSiam, self).__init__()


        # Same as SimSiam
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, pretrained=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.concat_feat = nn.Sequential(nn.Linear(dim * 2, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(prev_dim, dim))  # output layer

        # build a 2-layer predictor for paired data
        self.predictor = nn.Sequential(nn.Linear(dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(prev_dim, dim)) # output layer
        self.out_dim = dim

    def forward(self, x1, y1 , x2, y2):
        """
        Input:
            x1, y1: first views of pair of images
            x2, y2: second views of pair of images images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        # compute features for one view for paired images
        z1_x = self.encoder(x1) # NxC
        z1_y = self.encoder(y1)  # NxC
        z1 = torch.cat((z1_x, z1_y), dim=1) # N x 2C

        z2_x = self.encoder(x2)  # NxC
        z2_y = self.encoder(y2)  # NxC
        z2 = torch.cat((z2_x, z2_y), dim=1)  # N x 2C

        z1 = self.concat_feat(z1)
        z2 = self.concat_feat(z2)

        p1 = self.predictor(z1) # NxC'
        #p2 = self.predictor(z2) # NxC'

        return p1, z2.detach()
