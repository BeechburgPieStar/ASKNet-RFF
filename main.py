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

from utils.load_data import load_single_dataset
from backbones.PatchNet import PatchNet


def get_k_stats(model):
    if not hasattr(model, "cutoff"):
        return None
    logits = model.cutoff.selection_logits.detach().float().cpu()
    cands = torch.tensor(model.cutoff.candidates, dtype=torch.float)
    probs = torch.softmax(logits, dim=0)
    argmax_idx = int(logits.argmax().item())
    argmax_k = int(cands[argmax_idx].item())
    soft_k = float((probs * cands).sum().item())
    top_prob = float(probs[argmax_idx].item())
    return argmax_k, soft_k, top_prob


def format_k_summary(k_history, candidates):
    lines = [
        "=" * 72,
        "Per-epoch k selection summary",
        f"Candidates: {list(candidates)}",
        "-" * 72,
        "{:>5} | {:>8} | {:>10} | {:>10}".format("epoch", "argmax_k", "soft_k", "top_prob"),
        "-" * 72,
    ]
    for rec in k_history:
        lines.append("{:>5d} | {:>8d} | {:>10.3f} | {:>10.4f}".format(
            rec["epoch"], rec["argmax_k"], rec["soft_k"], rec["top_prob"]))
    lines.append("-" * 72)
    if k_history:
        final = k_history[-1]
        lines.append(f"Final selected k (last epoch): {final['argmax_k']}")
        most_common_k, freq = Counter(r["argmax_k"] for r in k_history).most_common(1)[0]
        lines.append(f"Most frequent k across epochs: {most_common_k} "
                     f"({freq}/{len(k_history)} epochs)")
    lines.append("=" * 72)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser("ASKNet")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--patience_early_stop", type=int, default=10)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--dataset_name", type=str, default="ManyRx")
    parser.add_argument("--exp", type=str, default="CRD", choices=["CRD", "CR"])
    parser.add_argument("--train_date", type=int, nargs="+", default=[1, 2])

    parser.add_argument("--all_test_round", type=int, default=4)
    parser.add_argument("--test_round", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])

    parser.add_argument("--code_state", type=str, default="train_test",
                        choices=["only_train", "only_test", "train_test"])
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_receivers(all_num=12, all_test_round=4, test_round=0):
    if not (0 <= test_round < all_test_round):
        raise ValueError(f"test_round {test_round} not in [0, {all_test_round - 1}]")
    if all_num % all_test_round != 0:
        raise ValueError(f"receiver count {all_num} not divisible by {all_test_round} rounds")

    receivers = list(range(all_num))
    per_round = all_num // all_test_round
    start = test_round * per_round
    end = all_num if test_round == all_test_round - 1 else start + per_round
    test = receivers[start:end]
    train = [r for r in receivers if r not in test]
    return train, test


def prepare_dataset(dataset_name, rx_indexes, date_indexes, tx_num, is_train, seed):
    x_all, y_all = [], []
    for rx_index in rx_indexes:
        for date_index in date_indexes:
            x, y = load_single_dataset(dataset_name, rx_index, date_index, tx_num, "non_equalized")
            x_all.append(x)
            y_all.append(y)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    if not is_train:
        return x_all, y_all

    indices = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=seed)
    return (x_all[train_idx], y_all[train_idx]), (x_all[val_idx], y_all[val_idx])


def _to_device(data, target, device):
    return data.to(device).float(), target.to(device).long()


def train_epoch(model, criterion, train_loader, optimizer, epoch, device, grad_clip=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in train_loader:
        data, target = _to_device(data, target, device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Train Epoch {epoch}\tLoss: {avg_loss:.6f}, Acc: {correct}/{total} ({acc:.2f}%)")
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
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Validation\tLoss: {avg_loss:.4f}, Acc: {correct}/{total} ({acc:.2f}%)")
    return avg_loss, acc


@torch.no_grad()
def test_epoch(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    for data, target in test_loader:
        data, target = _to_device(data, target, device)
        logits = model(data)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += data.size(0)

    acc = correct / max(total, 1)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def train_and_evaluate(model, train_loader, val_loader, epochs, save_path,
                       lr=1e-3, weight_decay=0.0, patience_early_stop=10,
                       scheduler_patience=5, min_lr=1e-6, grad_clip=None,
                       device="cuda", log_path=None):
    device = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                      patience=scheduler_patience, verbose=True, min_lr=min_lr)
    except TypeError:

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                      patience=scheduler_patience, min_lr=min_lr)

    best_val_loss = float("inf")
    no_improve = 0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    k_history = []
    init_stats = get_k_stats(model)
    if init_stats is not None:
        argmax_k, soft_k, top_prob = init_stats
        k_history.append({"epoch": 0, "argmax_k": argmax_k, "soft_k": soft_k,
                          "top_prob": top_prob, "train_loss": None, "val_loss": None})
        print(f"[k@init] argmax_k={argmax_k}  soft_k={soft_k:.3f}  top_prob={top_prob:.4f}")

    for epoch in range(1, epochs + 1):
        train_loss, _ = train_epoch(model, criterion, train_loader, optimizer,
                                    epoch, device, grad_clip=grad_clip)
        val_loss, _ = evaluate_epoch(model, criterion, val_loader, epoch, device)
        scheduler.step(val_loss)

        stats = get_k_stats(model)
        if stats is not None:
            argmax_k, soft_k, top_prob = stats
            k_history.append({"epoch": epoch, "argmax_k": argmax_k, "soft_k": soft_k,
                              "top_prob": top_prob, "train_loss": float(train_loss),
                              "val_loss": float(val_loss)})
            print(f"[k@epoch {epoch}] argmax_k={argmax_k}  soft_k={soft_k:.3f}  "
                  f"top_prob={top_prob:.4f}")

        if val_loss < best_val_loss:
            print(f"Validation loss improved {best_val_loss:.6f} -> {val_loss:.6f}, saving model.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s).")

        if no_improve >= patience_early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break
        print("-" * 48)

    if k_history:
        cutoff = getattr(model, "cutoff", None)
        candidates = cutoff.candidates if cutoff is not None else []
        summary = format_k_summary(k_history, candidates)
        print("\n" + summary + "\n")

        if log_path is not None:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                f.write(summary + "\n")
            print(f"[k-trace] wrote {log_path}")

    return best_val_loss, k_history


def build_model(conf, num_classes):
    return PatchNet(
        candidates=conf.candidates,
        patch_size=conf.patch_size,
        embed_dim=conf.embed_dim,
        mlp_ratio=conf.mlp_ratio,
        num_classes=num_classes,
    )


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    setup_seed(conf.seed)

    if conf.dataset_name == "ManySig":
        tx_num, rx_num = 6, 12
    elif conf.dataset_name == "ManyRx":
        tx_num, rx_num = 10, 32
    else:
        raise ValueError(f"Unsupported dataset_name: {conf.dataset_name}")

    rx_train, rx_test = split_receivers(rx_num, conf.all_test_round, conf.test_round)
    print(f"Train receivers: {rx_train}")
    print(f"Test receivers:  {rx_test}")

    (x_train, y_train), (x_val, y_val) = prepare_dataset(
        conf.dataset_name, rx_train, conf.train_date, tx_num, True, conf.seed)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=conf.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val),
                            batch_size=conf.batch_size, shuffle=False, drop_last=False)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train_dates_str = "_".join(map(str, conf.train_date))
    tag = (f"{conf.dataset_name}_{conf.exp}_date{train_dates_str}_round{conf.test_round}_"
           f"seed{conf.seed}_ps{conf.patch_size}_d{conf.embed_dim}_mlp{conf.mlp_ratio}")
    save_path = f"weights/{tag}.pth"
    log_path = f"logs/{tag}.k_trace.txt"
    print(f"[ckpt] {save_path}")
    print(f"[log ] {log_path}")

    if conf.code_state in ("only_train", "train_test"):
        model = build_model(conf, num_classes=tx_num)
        train_and_evaluate(
            model, train_loader, val_loader,
            epochs=conf.epochs, save_path=save_path,
            lr=conf.lr, weight_decay=conf.wd,
            patience_early_stop=conf.patience_early_stop,
            scheduler_patience=conf.scheduler_patience,
            min_lr=conf.min_lr, grad_clip=conf.grad_clip,
            device="cuda", log_path=log_path)

    if conf.code_state in ("only_test", "train_test"):
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_test = build_model(conf, num_classes=tx_num)
        model_test.load_state_dict(torch.load(save_path, map_location="cpu"))
        model_test = model_test.to(device)

        final_stats = get_k_stats(model_test)
        if final_stats is not None:
            fa_k, fs_k, fp = final_stats
            print(f"[k@test] argmax_k={fa_k}  soft_k={fs_k:.3f}  top_prob={fp:.4f}")

        if conf.exp == "CR":
            test_dates = conf.train_date
        else:
            test_dates = [d for d in (1, 2, 3, 4) if d not in conf.train_date]
        x_test, y_test = prepare_dataset(conf.dataset_name, rx_test, test_dates,
                                         tx_num, False, conf.seed)

        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_loader = DataLoader(TensorDataset(x_test, y_test),
                                 batch_size=32, shuffle=False, drop_last=False)

        test_acc = test_epoch(model_test, test_loader, device)

        try:
            with open(log_path, "a") as f:
                f.write("\n[test]\n")
                f.write(f"dataset  : {conf.dataset_name}\n")
                f.write(f"exp      : {conf.exp}\n")
                f.write(f"round    : {conf.test_round}\n")
                if final_stats is not None:
                    f.write(f"test_k   : {fa_k}  (soft_k={fs_k:.3f}, top_prob={fp:.4f})\n")
                f.write(f"test_acc : {test_acc:.4f}\n")
        except OSError as e:
            print(f"[warn] could not append test result to log: {e}")


if __name__ == "__main__":
    main()