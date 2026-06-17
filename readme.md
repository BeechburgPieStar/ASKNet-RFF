# ASKNet: Adaptive Spectral Koopman Network for Cross-Domain Radio Frequency Fingerprinting

Official PyTorch implementation of the IEEE Communications Letters paper
**"ASKNet: Adaptive Spectral Koopman Network for Cross-Domain Radio Frequency Fingerprinting."**

ASKNet tackles cross-domain RFF under **joint channel and receiver variations**. A spectral analysis shows that, in the low-frequency region, channel/receiver distortions reduce to a *uniform gain* plus a *linear phase*, both of which preserve the device fingerprint. ASKNet exploits this with two lightweight modules:

- **Learnable Spectral Gating (LSG)** — a data-driven, end-to-end-optimized low-frequency cutoff (Gumbel-Softmax selection over a candidate band set).
- **Spectral Koopman Operator (SKO)** — a unitary `K = U·diag(e^{jφ})·Uᴴ` that compensates the residual linear phase without amplifying energy.

The refined signal is then classified by a compact patch-based head. ASKNet outperforms representative domain-generalization and augmentation baselines on **WiSig** and **LoRa** with one-to-two orders of magnitude fewer parameters and FLOPs.

---

## Repository structure

```
ASKNet-RFF/
├── WiSig/                      # WiSig experiments (ManySig / ManyRx)
│   ├── main.py                 # train / test entry point
│   ├── backbones/
│   │   └── PatchNet.py         # LSG + SKO + patch-based classifier
│   ├── utils/
│   │   └── load_data.py        # .pkl loader, per-sample power normalization
│   ├── dataset/                # <-- put WiSig data here (see below)
│   ├── logs/                   # per-epoch k-trace and test logs
│   └── weights/                # checkpoints (created at runtime)
│
└── Lora/                       # LoRa experiments (CIS + ASK refinement)
    ├── run_CIS.py              # train / test entry point
    ├── backbones/
    │   └── ResCISNet.py        # ASKModule + channel-independent spectrogram + ResNet
    ├── utils/
    │   └── load_data_h5.py     # .h5 loader
    ├── dataset/                # <-- put LoRa data here (see below)
    ├── logs/
    └── weights/
```

## Environment

The results in the paper were obtained with:

| Component | Version |
|-----------|---------|
| Python    | 3.8.5 |
| PyTorch   | 1.11.0 (CUDA 11.3) |
| GPU       | NVIDIA RTX 3080 Ti |

Minimal dependencies:

```bash
pip install torch numpy scikit-learn h5py
```

> The code also runs on newer PyTorch versions; the scripts already handle the
> removal of the `verbose` argument in `ReduceLROnPlateau` (PyTorch ≥ 2.3).
> `h5py` is only required for the LoRa experiments.

## Data and pretrained weights

The datasets (preprocessed) and pretrained checkpoints are shared via Baidu Netdisk:

- **Link:** https://pan.baidu.com/s/17DIUaGsYrdiJjroAQ2fQyg
- **Code:** `peu9`

After downloading, place the files so the directory layout matches the loaders.

### WiSig

`WiSig/utils/load_data.py` expects per-(receiver, day) pickle files:

```
WiSig/dataset/<Subset>/non_equalized/date<d>/rx_<rx-id>_data.pkl
```

where `<Subset>` is `ManySig` or `ManyRx`, `<d> ∈ {1,2,3,4}`, and each `.pkl`
holds a dict with `data[tx_index]` of shape `(N, L, 2)` (the loader transposes
to `(N, 2, L)`, keeps the first 100 samples per Tx, and applies per-sample power
normalization). Receiver ids follow `RX_INDEXES_MANYSIG` / `RX_INDEXES_MANYRX`
defined in the loader.

| Subset  | Transmitters | Receivers | Sample length |
|---------|:------------:|:---------:|:-------------:|
| ManySig | 6            | 12        | 256           |
| ManyRx  | 10           | 32        | 256           |

### LoRa

`Lora/utils/load_data_h5.py` expects HDF5 files:

```
Lora/dataset/Train/<rx>_day<d>_wireless_train.h5
Lora/dataset/Test/<rx>_day<d>_wireless_test.h5
```

where `<rx> ∈ {n210_1, rtl_6}`. Each `.h5` stores `data` of shape `(N, 2L)`
(concatenated I/Q) and `label`. The last `last_k_classes` devices are kept and
re-labeled to `[0, last_k_classes-1]`.

## Quick start

### WiSig (ManySig / ManyRx)

A leave-one-group-out cross-validation over four receiver groups is used. Run
each of the four rounds (`--test_round 0..3`) and average the accuracies.

```bash
cd WiSig

# ManySig, cross receiver + cross day (CRD), round 0
python main.py --dataset_name ManySig --exp CRD --test_round 0 --seed 2023

# loop over all four rounds
for r in 0 1 2 3; do
    python main.py --dataset_name ManySig --exp CRD --test_round $r --seed 2023
done

# ManyRx
for r in 0 1 2 3; do
    python main.py --dataset_name ManyRx --exp CRD --test_round $r --seed 2023
done
```

Key arguments (`main.py`):

| Argument         | Default                  | Description |
|------------------|--------------------------|-------------|
| `--dataset_name` | `ManyRx`                 | `ManySig` or `ManyRx` |
| `--exp`          | `CRD`                    | `CRD` (cross receiver + day) or `CR` (cross receiver only) |
| `--train_date`   | `1 2`                    | training days; CRD tests on the remaining days |
| `--all_test_round` / `--test_round` | `4` / `0`  | LOGO cross-validation control |
| `--candidates`   | `8 16 24 32 48 64`       | LSG candidate half-bandwidths |
| `--patch_size`   | `64`                     | classifier patch length |
| `--embed_dim`    | `128`                    | token embedding dimension |
| `--code_state`   | `train_test`             | `only_train` / `only_test` / `train_test` |

Checkpoints go to `WiSig/weights/`, and the per-epoch LSG cutoff trace + final
test accuracy go to `WiSig/logs/`.

### LoRa (CIS + ASKNet refinement)

Following Fig. 3 in the paper, the model is trained on N210 (Day 1) and tested
on RTL across the later days:

```bash
cd Lora

python run_CIS.py \
    --train_rx n210_1 --train_day 1 \
    --test_rx  rtl_6  --test_day 2 3 4 \
    --last_k_classes 10 --per_class 800 --seq_len 8192 \
    --seed 2023
```

`run_CIS.py` reports per-day and average accuracy and writes the k-trace to
`Lora/logs/`.

## Reusing the signal-refinement module

The core contribution — LSG + SKO — is self-contained and framework-agnostic.
In the LoRa code it is wrapped as `ASKModule` (`Lora/backbones/ResCISNet.py`):
it takes a `[B, 2, L]` I/Q tensor and returns a refined `[B, 2, L]` tensor, so it
can be dropped in front of any existing RFF backbone:

```python
from backbones.ResCISNet import ASKModule

ask = ASKModule(seq_len=8192, candidates=[128, 256, 512, 1024, 2048, 4096])
x_refined = ask(x_iq)        # x_iq: [B, 2, L]  ->  x_refined: [B, 2, L]
```

## Acknowledgments

This work builds on two publicly available datasets, and we gratefully
acknowledge their authors:

- **WiSig** — S. Hanna, S. Karunaratne, and D. Cabric, "WiSig: A Large-Scale
  WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,"
  *IEEE Access*, vol. 10, pp. 22808–22818, 2022.
- **LoRa** — G. Shen, J. Zhang, A. Marshall, et al., "Towards Receiver-Agnostic
  and Collaborative Radio Frequency Fingerprint Identification," *IEEE
  Transactions on Mobile Computing*, vol. 23, no. 7, pp. 7618–7634, 2024.

The LoRa pipeline also reuses the channel-independent spectrogram (CIS)
representation introduced in the work above. Please cite these datasets if you
use them through this repository.
