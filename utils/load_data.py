import os
import pickle

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


RX_INDEXES_MANYSIG = [
    "1-1", "1-19", "2-1", "2-19", "3-19", "7-7", "7-14",
    "8-8", "14-7", "18-2", "19-2", "20-1",
]

RX_INDEXES_MANYRX = [
    "1-1", "1-19", "1-20", "2-1", "2-19", "3-19",
    "7-7", "7-14", "8-7", "8-8", "8-14",
    "13-7", "13-14", "14-7",
    "18-2", "18-19",
    "19-1", "19-2", "19-19", "19-20",
    "20-1", "20-19", "20-20",
    "23-1", "23-3", "23-5", "23-6", "23-7",
    "24-5", "24-6", "24-13", "24-16",
]

RX_INDEXES = {
    "ManySig": RX_INDEXES_MANYSIG,
    "ManyRx": RX_INDEXES_MANYRX,
}


def preprocessing(x):
    for i in range(x.shape[0]):
        power = np.sum(x[i, 0, :] ** 2 + x[i, 1, :] ** 2) / x.shape[2]
        x[i] = x[i] / np.sqrt(power)
    return x


def load_single_dataset(dataset, rx_index, date_index, tx_num, is_eq):
    if dataset not in RX_INDEXES:
        raise ValueError(f"Unknown dataset: {dataset}")
    rx_indexes = RX_INDEXES[dataset]

    folder_path = os.path.join(CURRENT_DIR, "..", "dataset", dataset, is_eq)
    file_path = os.path.join(folder_path, f"date{date_index}",
                             f"rx_{rx_indexes[rx_index]}_data.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    x_list, y_list = [], []
    for tx_index in range(tx_num):
        tx_data = data["data"][tx_index]
        tx_data = np.transpose(tx_data, (0, 2, 1))[:100]
        x_list.append(tx_data)
        y_list.extend([tx_index] * tx_data.shape[0])

    x = preprocessing(np.concatenate(x_list, axis=0))
    y = np.array(y_list)
    return x, y