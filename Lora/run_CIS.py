# -*- coding: utf-8 -*-
import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from utils.load_data_h5 import prepare_dataset_h5, prepare_test_by_day
from backbones.ResCISNet import ResCISNet


# ---------------------------------------------------------------------
# Per-epoch k tracking helpers
# ---------------------------------------------------------------------
def get_k_stats(model):
    """Return (argmax_k, soft_k, top_prob) from the ASK LearnableCutoff."""
    if not hasattr(model, "ask") or not hasattr(model.ask, "cutoff"):
        return None
    logits = model.ask.cutoff.selection_logits.detach().float().cpu()
    cands = torch.tensor(model.ask.cutoff.candidates, dtype=torch.float)
    probs = torch.softmax(logits, dim=0)
    argmax_idx = int(logits.argmax().item())
    argmax_k = int(cands[argmax_idx].item())
    soft_k = float((probs * cands).sum().item())
    top_prob = float(probs[argmax_idx].item())
    return argmax_k, soft_k, top_prob


def format_k_summary(k_history, candidates):
    lines = []
    lines.append("=" * 72)
    lines.append("Per-epoch k selection summary")
    lines.append("Candidates: " + str(list(candidates)))
    lines.append("-" * 72)
    lines.append("{:>5} | {:>10} | {:>12} | {:>10}".format(
        "epoch", "argmax_k", "soft_k", "top_prob"))
    lines.append("-" * 72)
    for rec in k_history:
        lines.append("{:>5d} | {:>10d} | {:>12.3f} | {:>10.4f}".format(
            rec["epoch"], rec["argmax_k"], rec["soft_k"], rec["top_prob"]))
    lines.append("-" * 72)
    if k_history:
        final = k_history[-1]
        lines.append("Final selected k (argmax, last epoch): {}".format(final["argmax_k"]))
        cnt = Counter(r["argmax_k"] for r in k_history)
        most_common_k, freq = cnt.most_common(1)[0]
        lines.append("Most frequently selected k across epochs: {} (hit {}/{} epochs)".format(
            most_common_k, freq, len(k_history)))
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Args / seed
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("ResCISNet")

    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None)

    parser.add_argument('--patience_early_stop', type=int, default=20)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--dataset_dirname', type=str, default="dataset")
    parser.add_argument('--last_k_classes', type=int, default=10)
    parser.add_argument('--per_class', type=int, default=800)
    parser.add_argument('--seq_len', type=int, default=8192)

    parser.add_argument('--train_rx', type=str, nargs='+', default=["rtl_6"],
                        choices=["n210_1", "rtl_6"])
    parser.add_argument('--test_rx', type=str, nargs='+', default=["n210_1"],
                        choices=["n210_1", "rtl_6"])

    parser.add_argument('--train_day', type=int, nargs='+', default=[1])
    parser.add_argument('--test_day', type=int, nargs='+', default=[1, 2, 3, 4])

    parser.add_argument('--win_len', type=int, default=128)
    parser.add_argument('--crop_ratio', type=float, default=0.3)

    # candidates: max = seq_len / 2 is the physical limit for complex baseband.
    parser.add_argument('--candidates', type=int, nargs='+',
                        default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument('--tau_start', type=float, default=5.0)
    parser.add_argument('--tau_end', type=float, default=0.1)

    parser.add_argument('--code_state', type=str, default="train_test",
                        choices=["only_train", "only_test", "train_test"])
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _to_device(data, target, device):
    data = data.to(device).float()
    target = target.to(device).long()
    return data, target


def get_tau(epoch, total_epochs, tau_start=5.0, tau_end=0.1):
    ratio = min(epoch / max(total_epochs, 1), 1.0)
    return tau_start + (tau_end - tau_start) * ratio


def train_epoch(model, criterion, train_loader, optimizer, epoch, device,
                grad_clip=None, tau=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in train_loader:
        data, target = _to_device(data, target, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data, tau=tau)
        loss = criterion(logits, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print("Train Epoch: {}\tLoss: {:.6f}, Acc: {}/{} ({:.2f}%), tau={:.3f}".format(
        epoch, avg_loss, correct, total, acc, tau))
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(model, criterion, val_loader, epoch, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in val_loader:
        data, target = _to_device(data, target, device)
        logits = model(data)
        loss = criterion(logits, target)

        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print("\nValidation set: Loss: {:.4f}, Acc: {}/{} ({:.2f}%)\n".format(
        avg_loss, correct, total, acc))
    return avg_loss, acc


@torch.no_grad()
def test_epoch(model, test_loader, device, tag="Test"):
    model.eval()
    correct, total = 0, 0
    for data, target in test_loader:
        data, target = _to_device(data, target, device)
        logits = model(data)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    acc = correct / max(total, 1)
    print("{} Accuracy: {:.2f}%".format(tag, acc * 100))
    return acc


def train_and_evaluate(
    model, train_loader, val_loader, epochs, save_path,
    lr=1e-3, weight_decay=0.0, patience_early_stop=10,
    scheduler_patience=5, min_lr=1e-6, grad_clip=None,
    device="cuda", tau_start=5.0, tau_end=0.1, log_path=None,
):
    device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss().to(device)
    try:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1,
            patience=scheduler_patience, verbose=True, min_lr=min_lr,
        )
    except TypeError:
        # PyTorch >= 2.3 removed the `verbose` kwarg
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1,
            patience=scheduler_patience, min_lr=min_lr,
        )

    best_val_loss = float('inf')
    no_improve = 0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    k_history = []
    init_stats = get_k_stats(model)
    if init_stats is not None:
        ak, sk, tp = init_stats
        k_history.append({
            "epoch": 0, "argmax_k": ak, "soft_k": sk, "top_prob": tp,
            "train_loss": None, "val_loss": None,
        })
        print("[k@init]  argmax_k={}  soft_k={:.3f}  top_prob={:.4f}".format(ak, sk, tp))

    for epoch in range(1, epochs + 1):
        tau = get_tau(epoch, epochs, tau_start, tau_end)
        tr_loss, _ = train_epoch(model, criterion, train_loader, optimizer, epoch,
                                 device, grad_clip=grad_clip, tau=tau)
        val_loss, _ = evaluate_epoch(model, criterion, val_loader, epoch, device)
        scheduler.step(val_loss)

        stats = get_k_stats(model)
        if stats is not None:
            ak, sk, tp = stats
            k_history.append({
                "epoch": epoch, "argmax_k": ak, "soft_k": sk, "top_prob": tp,
                "train_loss": float(tr_loss), "val_loss": float(val_loss),
            })
            print("[k@epoch {:d}]  argmax_k={}  soft_k={:.3f}  top_prob={:.4f}".format(
                epoch, ak, sk, tp))

        if val_loss < best_val_loss:
            print("Validation loss improved {:.6f} -> {:.6f}. Saving model...".format(
                best_val_loss, val_loss))
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            print("No improvement for {} epoch(s).".format(no_improve))

        if no_improve >= patience_early_stop:
            print("Early stopping at epoch {}.".format(epoch))
            break

        print("------------------------------------------------")

    if k_history:
        candidates = model.ask.cutoff.candidates if hasattr(model, "ask") else []
        summary = format_k_summary(k_history, candidates)
        print("\n" + summary + "\n")
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                f.write(summary + "\n")
            print("[k-trace] wrote {}".format(log_path))

    return best_val_loss, k_history


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    setup_seed(conf.seed)

    proj_root = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(proj_root, conf.dataset_dirname)

    print("Dataset root: {}".format(dataset_root))
    print("Train rx={}, day={}".format(conf.train_rx, conf.train_day))
    print("Test  rx={}, day={}".format(conf.test_rx, conf.test_day))
    print("last_k_classes={}, per_class={}, seq_len={}".format(
        conf.last_k_classes, conf.per_class, conf.seq_len))
    print("ASK candidates={}".format(conf.candidates))

    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    save_name = (
        "h5_rescisnet_"
        "k{}_per{}_"
        "trainrx{}_day{}_"
        "seed{}.pth".format(
            conf.last_k_classes, conf.per_class,
            "_".join(conf.train_rx), "_".join(map(str, conf.train_day)),
            conf.seed,
        )
    )
    save_path = os.path.join(save_dir, save_name)
    log_path = os.path.join("logs", save_name.replace(".pth", ".k_trace.txt"))
    print("Save path: {}".format(save_path))
    print("Log  path: {}".format(log_path))

    # --- Train ---
    if conf.code_state in ["only_train", "train_test"]:
        x_all, y_all = prepare_dataset_h5(
            dataset_root=dataset_root, split="Train",
            rx_list=conf.train_rx, day_list=conf.train_day,
            last_k_classes=conf.last_k_classes, per_class=conf.per_class,
            seq_len=conf.seq_len, verbose=True,
        )

        indices = np.arange(len(y_all))
        train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=conf.seed)

        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  batch_size=conf.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(x_val, y_val),
                                batch_size=conf.batch_size, shuffle=False, drop_last=False)

        model = ResCISNet(
            num_classes=conf.last_k_classes,
            win_len=conf.win_len,
            crop_ratio=conf.crop_ratio,
            seq_len=conf.seq_len,
            candidates=conf.candidates,
        )

        total_params = sum(p.numel() for p in model.parameters())
        ask_params = sum(p.numel() for p in model.ask.parameters())
        print("\nTotal params: {:,}".format(total_params))
        print("ASK  params:  {:,}".format(ask_params))
        print("Other params: {:,}\n".format(total_params - ask_params))

        train_and_evaluate(
            model, train_loader, val_loader,
            epochs=conf.epochs, save_path=save_path,
            lr=conf.lr, weight_decay=conf.wd,
            patience_early_stop=conf.patience_early_stop,
            scheduler_patience=conf.scheduler_patience,
            min_lr=conf.min_lr, grad_clip=conf.grad_clip,
            device="cuda",
            tau_start=conf.tau_start, tau_end=conf.tau_end,
            log_path=log_path,
        )

    # --- Test ---
    if conf.code_state in ["only_test", "train_test"]:
        model_test = ResCISNet(
            num_classes=conf.last_k_classes,
            win_len=conf.win_len,
            crop_ratio=conf.crop_ratio,
            seq_len=conf.seq_len,
            candidates=conf.candidates,
        )
        if not os.path.exists(save_path):
            raise IOError("Checkpoint not found: " + save_path)
        state = torch.load(save_path, map_location="cpu")
        model_test.load_state_dict(state)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_test = model_test.to(device)

        final_stats = get_k_stats(model_test)
        if final_stats is not None:
            fa_k, fs_k, fp = final_stats
            print("\n[k@test-ckpt]  argmax_k={}  soft_k={:.3f}  top_prob={:.4f}".format(
                fa_k, fs_k, fp))
        print("[ASK] Final selected k = {}".format(model_test.get_current_k()))

        day_to = prepare_test_by_day(
            dataset_root=dataset_root,
            rx_list=conf.test_rx, day_list=conf.test_day,
            last_k_classes=conf.last_k_classes, per_class=conf.per_class,
            seq_len=conf.seq_len, verbose=True,
        )

        print("\n========== Test Results ==========")
        results = {}
        for day in conf.test_day:
            x_te, y_te = day_to[day]
            x_te = torch.tensor(x_te, dtype=torch.float32)
            y_te = torch.tensor(y_te, dtype=torch.long)
            te_loader = DataLoader(TensorDataset(x_te, y_te),
                                   batch_size=512, shuffle=False, drop_last=False)
            acc = test_epoch(model_test, te_loader, device, tag="Day{}".format(day))
            results["Day{}".format(day)] = acc

        print("\n========== Summary ==========")
        for k, v in results.items():
            print("  {}: {:.2f}%".format(k, v * 100))
        avg_acc = float(np.mean(list(results.values())))
        print("  Average: {:.2f}%".format(avg_acc * 100))

        try:
            with open(log_path, "a") as f:
                f.write("\n[test]\n")
                f.write("train_rx  : {}\n".format(conf.train_rx))
                f.write("test_rx   : {}\n".format(conf.test_rx))
                f.write("train_day : {}\n".format(conf.train_day))
                f.write("test_day  : {}\n".format(conf.test_day))
                if final_stats is not None:
                    f.write("test_k    : {}  (soft_k={:.3f}, top_prob={:.4f})\n".format(
                        fa_k, fs_k, fp))
                for k, v in results.items():
                    f.write("{:<9s} : {:.4f}\n".format(k, v))
                f.write("Average   : {:.4f}\n".format(avg_acc))
        except Exception as e:
            print("[warn] could not append test result to log: {}".format(e))

    print("\nDone.")


if __name__ == '__main__':
    main()