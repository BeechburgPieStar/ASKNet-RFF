import torch
import torch.nn as nn
import torch.nn.functional as F


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
        K = U @ D @ U.conj().T
        return x @ K.T


class LearnableCutoff(nn.Module):

    def __init__(self, L, candidates=None):
        super().__init__()
        assert L % 2 == 0, "L must be even so that fftshift places DC at L // 2"
        self.L = L
        self.center = L // 2
        self.candidates = candidates if candidates is not None else [8, 16, 24, 32, 48, 64]
        self.num_candidates = len(self.candidates)
        self.max_k = max(self.candidates)
        self.max_modes = 2 * self.max_k - 1

        self.selection_logits = nn.Parameter(torch.zeros(self.num_candidates))
        self.register_buffer("masks", self._build_masks(L))

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

        final_mask = torch.einsum("c,cl->l", weights, self.masks)
        Xf_masked = Xf_shifted * final_mask.unsqueeze(0)

        selected_k = sum(w.item() * k for w, k in zip(weights, self.candidates))
        return Xf_masked, selected_k


class PatchEmbed1D(nn.Module):

    def __init__(self, in_chans=2, embed_dim=128, patch_size=16):
        super().__init__()
        stride = max(1, patch_size // 2)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class MLPBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(self.norm(x))))
        return x + self.drop(self.fc2(h))


class PatchNet(nn.Module):

    def __init__(self, L=256, candidates=None, patch_size=64, embed_dim=128,
                 mlp_ratio=2.0, num_classes=6, dropout=0.0):
        super().__init__()
        assert L % 2 == 0, "L must be even"
        self.L = L
        if candidates is None:
            candidates = [8, 16, 24, 32, 48, 64]

        self.cutoff = LearnableCutoff(L, candidates)
        self.center = self.cutoff.center
        self.max_k = self.cutoff.max_k
        self.max_modes = self.cutoff.max_modes

        self.koopman = KoopmanLayer(n_atoms=self.max_modes)
        self.patch_embed = PatchEmbed1D(in_chans=2, embed_dim=embed_dim, patch_size=patch_size)
        self.mlp_block = MLPBlock(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_k=False, tau=1.0):
        B, C, L = x.shape
        c = self.center

        x_complex = torch.complex(x[:, 0], x[:, 1])
        Xf_shift = torch.fft.fftshift(torch.fft.fft(x_complex, dim=-1), dim=-1)

        Xf_masked, selected_k = self.cutoff(Xf_shift, tau=tau)

        Xf_low = Xf_masked[:, c - self.max_k + 1 : c + self.max_k]

        Xf_evolved = self.koopman(Xf_low)
        Xf_full_shift = torch.zeros(B, L, dtype=torch.cfloat, device=x.device)
        Xf_full_shift[:, c - self.max_k + 1 : c + self.max_k] = Xf_evolved

        x_recon = torch.fft.ifft(torch.fft.ifftshift(Xf_full_shift, dim=-1), dim=-1)
        energy = torch.sum(x_recon.abs().pow(2), dim=-1, keepdim=True)
        x_normalized = x_recon / (torch.sqrt(energy) + 1e-8)

        x_out = torch.stack([x_normalized.real, x_normalized.imag], dim=1)
        tokens = self.mlp_block(self.patch_embed(x_out))
        logits = self.cls_head(tokens.mean(dim=1))

        if return_k:
            return logits, selected_k
        return logits

    def get_current_k(self):
        idx = self.cutoff.selection_logits.argmax().item()
        return self.cutoff.candidates[idx]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 4, 256
    model = PatchNet(L=L, num_classes=6)

    model.train()
    x = torch.randn(B, 2, L)
    logits, k = model(x, return_k=True)
    print(f"[train] logits: {tuple(logits.shape)}, selected_k: {k}")

    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"[eval]  logits: {tuple(logits.shape)}, current_k: {model.get_current_k()}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")