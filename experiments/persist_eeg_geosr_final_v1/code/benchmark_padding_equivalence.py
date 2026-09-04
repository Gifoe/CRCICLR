"""Check explicit SAME padding against canonical Conv2d padding."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import audit_primitives as ap


class ExplicitSame(ap.VanillaEEGNet):
    def forward_features(self, x):
        v = x.unsqueeze(1)
        # Conv2d SAME for (1,64) on width 1000 is left=31,right=32.
        v = self.bn1(F.conv2d(F.pad(v, (31, 32, 0, 0)), self.temporal.weight, self.temporal.bias, self.temporal.stride, 0, self.temporal.dilation, self.temporal.groups))
        v = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(v)))))
        # Conv2d SAME for (1,16) is left=7,right=8.
        v = self.drop2(self.pool2(F.elu(self.bn3(self.point(F.conv2d(F.pad(v, (7, 8, 0, 0)), self.depth.weight, self.depth.bias, self.depth.stride, 0, self.depth.dilation, self.depth.groups))))))
        return self.embedding(v.flatten(1))


def main():
    for device in ([torch.device("cuda")] if torch.cuda.is_available() else [torch.device("cpu")]):
        torch.manual_seed(123)
        a = ap.VanillaEEGNet(62).to(device)
        b = ExplicitSame(62).to(device)
        b.load_state_dict(a.state_dict())
        a.train(); b.train()
        x = torch.randn(4, 62, 1000, device=device)
        torch.manual_seed(999); ya = a(x)
        torch.manual_seed(999); yb = b(x)
        print(device, "forward_equal", torch.equal(ya, yb), "max_abs", float((ya-yb).abs().max()))
        ga = torch.autograd.grad(ya.sum(), tuple(a.parameters()), retain_graph=False)
        gb = torch.autograd.grad(yb.sum(), tuple(b.parameters()), retain_graph=False)
        print(device, "grad_equal", all(torch.equal(x,y) for x,y in zip(ga,gb)), "max_grad_abs", max(float((x-y).abs().max()) for x,y in zip(ga,gb)))


if __name__ == "__main__":
    main()
