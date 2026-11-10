"""MSD_musicnn (PyTorch) matched to the keys/shapes of MSD_musicnn_optimized_fp16.pt.

Forward semantics follow the public musicnn-pytorch port (conv -> ReLU -> BN,
residual adds in the mid-end, interleaved max/mean pooling in the back-end).
Input: log-mel patch (B, T, 96). Output: logits (B, 50); apply sigmoid for tags.

UNVERIFIED numerically: shapes/keys are checked by `check_keys` (no torch needed),
but outputs must still be diffed against a reference implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 96
BN_KW = dict(eps=0.001, momentum=0.01)


class ConvReLUBN(nn.Module):
    # keys: <name>.conv.weight / <name>.bn.*  (wrapped one level deeper below)
    def __init__(self, out_ch, kernel):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, kernel, padding=0)
        self.bn = nn.BatchNorm2d(out_ch, **BN_KW)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class _Wrap(nn.Module):
    # produces keys like timbral_1.conv.conv.weight, timbral_1.conv.bn.weight
    def __init__(self, out_ch, kernel):
        super().__init__()
        self.conv = ConvReLUBN(out_ch, kernel)


class TimbralBlock(_Wrap):
    def __init__(self, mel_bins, out_ch):
        super().__init__(out_ch, (7, mel_bins))

    def forward(self, x):                      # x: (B,1,T,96)
        x = F.pad(x, (0, 0, 3, 3))             # pad time only -> T preserved
        x = self.conv(x)                       # (B,C,T,96-k+1)
        return x.max(dim=3).values             # (B,C,T)


class TemporalBlock(_Wrap):
    def __init__(self, k, out_ch):
        super().__init__(out_ch, (k, 1))
        self.k = k

    def forward(self, x):                      # x: (B,1,T,96)
        total = self.k - 1                     # TF-style 'same': extra pad on the right
        x = F.pad(x, (0, 0, total // 2, total - total // 2))
        x = self.conv(x)                       # (B,C,T,96)
        return x.max(dim=3).values             # (B,C,T)


class MidEnd(nn.Module):
    def __init__(self, in_ch=561, f=64):
        super().__init__()
        self.c1_conv = nn.Conv2d(1, f, (7, in_ch)); self.c1_bn = nn.BatchNorm2d(f, **BN_KW)
        self.c2_conv = nn.Conv2d(1, f, (7, f));     self.c2_bn = nn.BatchNorm2d(f, **BN_KW)
        self.c3_conv = nn.Conv2d(1, f, (7, f));     self.c3_bn = nn.BatchNorm2d(f, **BN_KW)

    @staticmethod
    def _step(x, conv, bn):                    # x: (B,T,C) -> (B,T,f)
        x = F.pad(x.unsqueeze(1), (0, 0, 3, 3))            # (B,1,T+6,C)
        x = bn(F.relu(conv(x)))                            # (B,f,T,1)
        return x.squeeze(3).transpose(1, 2)

    def forward(self, front):                  # front: (B,T,561)
        x1 = self._step(front, self.c1_conv, self.c1_bn)
        x2 = self._step(x1, self.c2_conv, self.c2_bn)
        res2 = x2 + x1
        x3 = self._step(res2, self.c3_conv, self.c3_bn)
        res3 = x3 + res2
        return torch.cat([front, x1, res2, res3], dim=2)   # (B,T,753)


class Backend(nn.Module):
    def __init__(self, in_ch=753, hidden=200, n_tags=50):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_ch * 2, **BN_KW)
        self.fc1 = nn.Linear(in_ch * 2, hidden)
        self.bn_fc1 = nn.BatchNorm1d(hidden, **BN_KW)
        self.fc2 = nn.Linear(hidden, n_tags)

    def forward(self, z):                      # z: (B,T,753)
        mx = z.max(dim=1).values
        mean = z.mean(dim=1)
        z = torch.stack([mx, mean], dim=2).flatten(1)      # interleaved [max_i, mean_i]
        z = self.bn_in(z)
        z = self.bn_fc1(F.relu(self.fc1(z)))
        return self.fc2(z)


class MusicnnMSD(nn.Module):
    def __init__(self, n_tags=50):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1, **BN_KW)
        self.timbral_1 = TimbralBlock(int(0.4 * N_MELS), int(1.6 * 128))   # 38, 204
        self.timbral_2 = TimbralBlock(int(0.7 * N_MELS), int(1.6 * 128))   # 67, 204
        self.temp_1 = TemporalBlock(128, int(1.6 * 32))                    # 51
        self.temp_2 = TemporalBlock(64, int(1.6 * 32))
        self.temp_3 = TemporalBlock(32, int(1.6 * 32))
        self.midend = MidEnd(561, 64)
        self.backend = Backend(753, 200, n_tags)

    def forward(self, x):                      # x: (B,T,96) log-mel
        x = self.bn_input(x.unsqueeze(1))
        feats = [self.timbral_1(x), self.timbral_2(x),
                 self.temp_1(x), self.temp_2(x), self.temp_3(x)]
        front = torch.cat(feats, dim=1).transpose(1, 2)    # (B,T,561)
        return self.backend(self.midend(front))


def load_msd_musicnn(path, half=False, device="cpu"):
    state = torch.load(path, map_location="cpu")
    model = MusicnnMSD()
    # fp16 tensors are cast to fp32 by load_state_dict; BN stats are already fp32
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing}\nunexpected={unexpected}")
    model.eval().to(device)
    return model.half() if half else model
