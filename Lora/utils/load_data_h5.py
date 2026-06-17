# =========================
# File: utils/load_data_h5.py
# =========================
import os
import numpy as np
import h5py


def resolve_h5_path(root_dir: str, split: str, rx: str, day: int) -> str:
    if split not in ["Train", "Test"]:
        raise ValueError("split must be 'Train' or 'Test'")
    if rx not in ["n210_1", "rtl_6"]:
        raise ValueError("rx must be 'n210_1' or 'rtl_6'")
    if split == "Train":
        name = f"{rx}_day{day}_wireless_train.h5"
    else:
        name = f"{rx}_day{day}_wireless_test.h5"
    return os.path.join(root_dir, split, name)


def _to_2ch_from_iq_concat(x: np.ndarray) -> np.ndarray:
    """
    h5 data: (N,2L) [I... , Q...]
    return: (N,2,L)
    """
    if x.ndim != 2 or x.shape[1] % 2 != 0:
        raise ValueError(f"Expect (N,2L), got {x.shape}")
    l = x.shape[1] // 2
    i = x[:, :l]
    q = x[:, l:]
    return np.stack([i, q], axis=1).astype(np.float32)


def preprocessing_power_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    x: (N,2,L)
    power normalize per sample
    """
    p = np.mean(x[:, 0, :] ** 2 + x[:, 1, :] ** 2, axis=1)
    s = np.sqrt(np.maximum(p, eps))[:, None, None]
    return x / s


def load_single_h5(
    h5_path: str,
    last_k_classes: int = 10,
    per_class: int = 200,
    pkt_range: slice = None,
    remap_labels: bool = True,
    do_norm: bool = True,
    verbose: bool = False,
):
    """
    Read one h5 and keep only last_k_classes (by sorted unique label id).
    Return:
      x: (N,2,L) float32
      y: (N,) int64 in [0,last_k_classes-1] if remap_labels=True
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"h5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "data" not in f or "label" not in f:
            raise KeyError(f"h5 must have 'data' and 'label'. keys={list(f.keys())}")
        data = np.array(f["data"][:])
        label = np.array(f["label"][:]).astype(int).reshape(-1)

    if label.min() == 1 and (0 not in set(label.tolist())):
        label = label - 1

    classes_all = np.sort(np.unique(label))
    classes_keep = classes_all[-int(last_k_classes):] if last_k_classes is not None else classes_all

    x2 = _to_2ch_from_iq_concat(data)

    x_list, y_list = [], []
    per_raw, per_used = {}, {}

    for cls in classes_keep.tolist():
        idx_all = np.where(label == cls)[0]
        per_raw[int(cls)] = int(len(idx_all))

        idx = idx_all
        if pkt_range is not None:
            idx = idx[pkt_range]
        if per_class is not None:
            idx = idx[:per_class]

        per_used[int(cls)] = int(len(idx))
        if len(idx) == 0:
            continue

        x_list.append(x2[idx])
        y_list.append(np.full(len(idx), int(cls), dtype=np.int64))

    if len(x_list) == 0:
        raise RuntimeError(f"No samples loaded from {h5_path}")

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if remap_labels:
        label_map = {int(cls): i for i, cls in enumerate(classes_keep.tolist())}
        y = np.array([label_map[int(v)] for v in y], dtype=np.int64)
    else:
        label_map = None
        y = y.astype(np.int64)

    if do_norm:
        x = preprocessing_power_norm(x)

    if verbose:
        print(f"[h5] {h5_path}")
        print(f"[h5] classes_in_file={len(classes_all)} keep={len(classes_keep)} keep_ids={classes_keep.tolist()}")
        print("[h5] per-class raw -> used")
        for cls in classes_keep.tolist():
            print(f"  class {int(cls):02d}: {per_raw.get(int(cls),0)} -> {per_used.get(int(cls),0)}")
        if remap_labels:
            print(f"[h5] remap enabled, label_map={label_map}")

    return x, y


def _crop_or_pad_np(x: np.ndarray, target_len: int):
    if target_len is None:
        return x
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[:, :, :target_len]
    pad = target_len - x.shape[-1]
    return np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")


def prepare_dataset_h5(
    dataset_root: str,
    split: str,
    rx_list,
    day_list,
    last_k_classes: int,
    per_class: int,
    seq_len: int,
    verbose: bool = False,
):
    x_all, y_all = [], []
    for rx in rx_list:
        for day in day_list:
            h5_path = resolve_h5_path(dataset_root, split, rx, day)
            x, y = load_single_h5(
                h5_path=h5_path,
                last_k_classes=last_k_classes,
                per_class=per_class,
                remap_labels=True,
                do_norm=True,
                verbose=verbose,
            )
            x = _crop_or_pad_np(x, seq_len)
            x_all.append(x)
            y_all.append(y)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return x_all, y_all


def prepare_test_by_day(
    dataset_root: str,
    rx_list,
    day_list,
    last_k_classes: int,
    per_class: int,
    seq_len: int,
    verbose: bool = False,
):
    day_to = {}
    for day in day_list:
        x, y = prepare_dataset_h5(
            dataset_root=dataset_root,
            split="Test",
            rx_list=rx_list,
            day_list=[day],
            last_k_classes=last_k_classes,
            per_class=per_class,
            seq_len=seq_len,
            verbose=verbose,
        )
        day_to[day] = (x, y)
    return day_to