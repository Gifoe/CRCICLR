import torch
import torch.nn.functional as F
from litebn_tfformer import LiteBNTFFormer


class CachedTFFormer(LiteBNTFFormer):
    """Exact TFFormer with an optional cache for only parameter-free STFT magnitudes."""
    def project_cached_stft(self, cached):
        out = []
        for index, value in enumerate(cached):
            value = self.spectral.proj[index](value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            value = F.adaptive_avg_pool2d(self.spectral.cnn[index](value), (24, 4)).permute(0, 2, 3, 1)
            value = value + self.spectral.scale[index] + self.spectral.freq[:, None] + self.spectral.time[None, :]
            out.append(value.reshape(value.shape[0], 96, 64))
        return torch.cat(out, dim=1)

    def forward(self, x, cached_stft=None):
        raw, value, branches = x, x.unsqueeze(1), []
        for temporal, temporal_bn, spatial, spatial_bn in zip(self.base.temporal, self.base.temporal_norm, self.base.spatial, self.base.spatial_norm):
            branch = F.avg_pool2d(F.elu(spatial_bn(spatial(F.elu(temporal_bn(temporal(value)))))), (1, 4))
            branches.append(F.dropout(branch, .2, self.base.training))
        value = torch.cat(branches, 1)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm1(self.base.point1(self.base.depth1(value)))), (1, 2)), .15, self.base.training)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm2(self.base.point2(self.base.depth2(value)))), (1, 2)), .15, self.base.training)
        tokens = value.squeeze(2).transpose(1, 2)
        spectral = self.spectral(raw) if cached_stft is None else self.project_cached_stft(cached_stft)
        tokens = self.g2(self.conv(self.g1(self.cross(tokens, spectral))))
        representation = self.base.drop(self.base.embedding(self.base.pool(tokens.transpose(1, 2).unsqueeze(2)).flatten(1)))
        return self.base.head(representation), representation
