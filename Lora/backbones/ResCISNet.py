# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== ASK (double-sided, fftshift) ========================

class KoopmanLayer(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.U = nn.Parameter(torch.randn(n_atoms, n_atoms, dtype=torch.cfloat))
        self.theta = nn.Parameter(torch.zeros(n_atoms))
        with torch.no_grad():
            self.U.data = torch.linalg.qr(self.U)[0]

    def forward(self, x):
        U, _ = torch.linalg.qr(self.U)
        D = torch.diag(torch.exp(1j * self.theta))
        K_mat = U @ D @ U.conj().T
        return x @ K_mat.T


class LearnableCutoff(nn.Module):
    def __init__(self, L, candidates=None):
        super().__init__()
        assert L % 2 == 0, "L must be even so fftshift places DC at L//2"
        self.L = L
        self.center = L // 2
        if candidates is None:
            candidates = [128, 256, 512, 1024, 2048, 4096]
        assert max(candidates) <= L // 2, (
            "max(candidates)=" + str(max(candidates)) +
            " must be <= L/2=" + str(L // 2) + " (physical limit)"
        )
        self.candidates = candidates
        self.num_candidates = len(candidates)
        self.max_k = max(candidates)
        self.max_modes = 2 * self.max_k - 1

        self.selection_logits = nn.Parameter(torch.zeros(self.num_candidates))
        self.register_buffer('masks', self._build_masks(L))

    def _build_masks(self, L):
        masks = torch.zeros(self.num_candidates, L)
        c = self.center
        for i, k in enumerate(self.candidates):
            masks[i, c - k + 1 : c + k] = 1.0
        return masks

    def forward(self, Xf_shifted, tau=1.0):
        if self.training:
            weights = F.gumbel_softmax(self.selection_logits, tau=tau, hard=True)
        else:
            idx = self.selection_logits.argmax()
            weights = F.one_hot(idx, self.num_candidates).float()

        final_mask = torch.einsum('c,cl->l', weights, self.masks)
        Xf_masked = Xf_shifted * final_mask.unsqueeze(0)
        selected_k = sum(w.item() * k for w, k in zip(weights, self.candidates))
        return Xf_masked, selected_k


class ASKModule(nn.Module):
    """ASK block for Lora:

    x [B, 2, L]
      -> FFT -> fftshift
      -> LearnableCutoff (gate on shifted spectrum)
      -> extract central (2*max_k - 1) contiguous bins
      -> Koopman evolve
      -> scatter back into shifted spectrum (other bins zero)
      -> ifftshift -> IFFT
      -> energy-normalize
      -> return [B, 2, L]
    """
    def __init__(self, seq_len=8192, candidates=None):
        super().__init__()
        assert seq_len % 2 == 0, "seq_len must be even"
        self.seq_len = seq_len

        if candidates is None:
            candidates = [128, 256, 512, 1024, 2048, 4096]

        self.cutoff = LearnableCutoff(seq_len, candidates)
        self.center = self.cutoff.center
        self.max_k = self.cutoff.max_k
        self.max_modes = self.cutoff.max_modes   # 2 * max_k - 1

        self.koopman = KoopmanLayer(n_atoms=self.max_modes)

    def forward(self, x, tau=1.0):
        B, C, L = x.shape

        # 1) FFT + fftshift
        x_complex = torch.complex(x[:, 0], x[:, 1])
        Xf = torch.fft.fft(x_complex, dim=-1)
        Xf_shift = torch.fft.fftshift(Xf, dim=-1)

        # 2) LSG gate on shifted spectrum
        Xf_masked, selected_k = self.cutoff(Xf_shift, tau=tau)

        # 3) extract central contiguous low-freq band of width (2*max_k - 1)
        c = self.center
        Xf_low = Xf_masked[:, c - self.max_k + 1 : c + self.max_k]

        # 4) Koopman evolution
        Xf_evolved = self.koopman(Xf_low)

        # 5) scatter back into shifted spectrum
        Xf_full_shift = torch.zeros(B, L, dtype=torch.cfloat, device=x.device)
        Xf_full_shift[:, c - self.max_k + 1 : c + self.max_k] = Xf_evolved

        # 6) ifftshift -> IFFT -> energy normalize
        Xf_full = torch.fft.ifftshift(Xf_full_shift, dim=-1)
        x_recon = torch.fft.ifft(Xf_full, dim=-1)
        energy = torch.sum(x_recon.abs().pow(2), dim=-1, keepdim=True)
        x_normalized = x_recon / (torch.sqrt(energy) + 1e-8)

        # 7) back to [B, 2, L]
        x_out = torch.stack([x_normalized.real, x_normalized.imag], dim=1)
        return x_out

    def get_current_k(self):
        idx = self.cutoff.selection_logits.argmax().item()
        return self.cutoff.candidates[idx]


# ======================== CIS + ResBlock (unchanged) ========================

class ChannelIndSpectrogramTorch(nn.Module):
    def __init__(self, win_len=128, crop_ratio=0.3, eps=1e-12):
        super().__init__()
        self.win_len = int(win_len)
        self.crop_ratio = float(crop_ratio)
        self.eps = float(eps)
        self.register_buffer("window", torch.ones(self.win_len, dtype=torch.float32))

    @staticmethod
    def _rms_norm_complex(xc, eps):
        amp2 = (xc.real ** 2 + xc.imag ** 2)
        rms = torch.sqrt(torch.mean(amp2, dim=-1, keepdim=True).clamp_min(eps))
        return xc / rms

    def forward(self, x2):
        if x2.ndim != 3 or x2.size(1) != 2:
            raise ValueError("Expect (B,2,L), got " + str(tuple(x2.shape)))
        xc = torch.complex(x2[:, 0, :], x2[:, 1, :])
        xc = self._rms_norm_complex(xc, self.eps)
        hop = self.win_len // 2
        spec = torch.stft(
            xc, n_fft=self.win_len, hop_length=hop, win_length=self.win_len,
            window=self.window, center=False, onesided=False, return_complex=True,
        )
        spec = torch.fft.fftshift(spec, dim=1)
        spec = spec + (self.eps + 0j)
        dspec = spec[:, :, 1:] / spec[:, :, :-1]
        amp = torch.abs(dspec)
        dspec_amp = torch.log10(amp * amp + self.eps)
        f0 = int(np.floor(self.win_len * self.crop_ratio))
        f1 = int(np.ceil(self.win_len * (1.0 - self.crop_ratio)))
        dspec_amp = dspec_amp[:, f0:f1, :]
        return dspec_amp.unsqueeze(1)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, first_layer=False):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=True)
        if first_layer or (in_ch != out_ch):
            self.short = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True)
        else:
            self.short = nn.Identity()

    def forward(self, x):
        fx = F.relu(self.conv1(x), inplace=True)
        fx = self.conv2(fx)
        sx = self.short(x)
        return F.relu(sx + fx, inplace=True)


# ======================== ResCISNet ========================

class ResCISNet(nn.Module):
    def __init__(self, num_classes, win_len=128, crop_ratio=0.3,
                 seq_len=8192, candidates=None):
        super().__init__()
        self.ask = ASKModule(seq_len=seq_len, candidates=candidates)
        self.cis = ChannelIndSpectrogramTorch(win_len=win_len, crop_ratio=crop_ratio)
        self.conv0 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True)
        self.rb1 = ResBlock2D(32, 32, k=3, first_layer=False)
        self.rb2 = ResBlock2D(32, 32, k=3, first_layer=False)
        self.rb3 = ResBlock2D(32, 64, k=3, first_layer=True)
        self.rb4 = ResBlock2D(64, 64, k=3, first_layer=False)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.adapt = nn.AdaptiveAvgPool2d((13, 16))
        self.fc = nn.Linear(64 * 13 * 16, 512)
        self.tx_fc = nn.Linear(512, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x2, tau=1.0):
        x2 = self.ask(x2, tau=tau)
        x = self.cis(x2)
        x = F.relu(self.conv0(x), inplace=True)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        feat = F.normalize(x, p=2, dim=1)
        x = F.relu(self.tx_fc(feat), inplace=True)
        logits = self.head(x)
        return logits

    def get_current_k(self):
        return self.ask.get_current_k()