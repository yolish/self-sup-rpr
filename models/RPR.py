import torch.nn as nn
import torch
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        x = F.gelu(self.fc_h(x))
        return self.fc_o(x)


class RPR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.rel_regressor_x =  PoseRegressor(encoder.out_dim, 3)
        self.rel_regressor_q = PoseRegressor(encoder.out_dim, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        self.encoder = encoder

    def forward(self, query, ref):
        out = self.encoder(query, ref)
        # regress relative x and q
        rel_x = self.rel_regressor_x(out)
        rel_q = self.rel_regressor_q(out)

        return {"rel_pose":torch.cat((rel_x,rel_q), dim=1)}






