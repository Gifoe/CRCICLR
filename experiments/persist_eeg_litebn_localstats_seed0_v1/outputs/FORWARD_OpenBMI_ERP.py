def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shallow = []
    x = x.unsqueeze(1)
    branches = []
    for (t, tn, s, sn) in zip(self.temporal, self.temporal_norm, self.spatial, self.spatial_norm):
        y = F.elu(tn(t(x)))
        y = F.elu(sn(s(y)))
        shallow.append(y)
        y = F.avg_pool2d(y, (1, 4))
        branches.append(F.dropout(y, 0.2, self.training))
    x = torch.cat(branches, 1)
    x = F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(x)))), (1, 2)), 0.15, self.training)
    x = F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(x)))), (1, 2)), 0.15, self.training)
    z = self.drop(self.local_embed(self.pool(x).flatten(1), shallow))
    return (self.head(z), z)
