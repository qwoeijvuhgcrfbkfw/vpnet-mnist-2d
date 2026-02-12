import os
from enum import Enum
from typing import Tuple

import torch
import pandas


class ImageDataType(Enum):
    MNIST_TRAIN = 0
    MNIST_TEST = 1


RANDOM_STATE = 42


def process_df(df: pandas.DataFrame, dataset_norm: Tuple[float, float] | None,
               device: torch.device) -> torch.utils.data.TensorDataset:
    data = torch.tensor(df.to_numpy(), device=device)

    labels, images = torch.split(data, [1, data.size(dim=1) - 1], dim=1)

    images = images / 255.0

    if dataset_norm is not None:
        images = (images - dataset_norm[0]) / dataset_norm[1]

    num_classes = torch.unique(labels).numel()

    if labels.shape[0] == 1:
        labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=num_classes).float().unsqueeze(dim=0)
    else:
        labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=num_classes).float()

    return torch.utils.data.TensorDataset(images, labels)


def load_image_data(data_type: ImageDataType, part: float = 1.0) -> torch.utils.data.TensorDataset | Tuple[
    torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    my_dir = os.path.dirname(os.path.abspath(__file__))

    paths = {
        ImageDataType.MNIST_TRAIN: os.path.join(my_dir, "../data/mnist-dataset/train_data/mnist_train.csv"),
        ImageDataType.MNIST_TEST: os.path.join(my_dir, "../data/mnist-dataset/test_data/mnist_test.csv"),
    }

    headers = {
        ImageDataType.MNIST_TRAIN: None,
        ImageDataType.MNIST_TEST: None,
    }

    norm = {
        ImageDataType.MNIST_TRAIN: (0.1307, 0.3081),
        ImageDataType.MNIST_TEST: (0.1307, 0.3081),
    }

    dataset_norm = norm[data_type]
    df = pandas.read_csv(paths[data_type], sep=",", header=headers[data_type])

    if part < 1.0:
        df_part = df.groupby(0, group_keys=False)[df.columns].apply(
            lambda x: x.sample(frac=part, random_state=RANDOM_STATE))
        df_rest = df.drop(df_part.index)

        df_part = df_part.reset_index(drop=True)
        df_rest = df_rest.reset_index(drop=True)

        return process_df(df_part, dataset_norm, device), process_df(df_rest, dataset_norm, device)

    return process_df(df, dataset_norm, device)
