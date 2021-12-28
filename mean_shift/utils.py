from typing import List

import numpy as np


def get_euclidean_distance(sample1: np.ndarray, sample2: np.ndarray) -> float:
    if len(sample1) != len(sample2):
        raise ValueError("Feature numbers of given samples are not equal.")

    return np.linalg.norm(sample1 - sample2)


def get_flat_kernel(
        cluster_center: np.ndarray, dataset: np.ndarray, kernel_parameters: List
) -> np.ndarray:
    diff_to_center, bandwidth = cluster_center - dataset, kernel_parameters[0]

    return np.array(
        [1. if num <= bandwidth else 0. for num in np.sqrt(np.power(diff_to_center, 2).sum(axis=1))]
    )


def get_gaussian_kernel(
        cluster_center: np.ndarray, dataset: np.ndarray, kernel_parameters: List
) -> np.ndarray:
    diff_to_center, beta = cluster_center - dataset, kernel_parameters[0]

    return np.exp(-1. * beta * np.power(diff_to_center, 2).sum(axis=1))


def get_truncated_gaussian_kernel(
        cluster_center: np.ndarray, dataset: np.ndarray, kernel_parameters: List
) -> np.ndarray:
    diff_to_center, bandwidth, beta = \
        cluster_center - dataset, kernel_parameters[0], kernel_parameters[1]
    norm_square = np.power(diff_to_center, 2).sum(axis=1)
    gaussian, bool_selector = np.exp(-1. * beta * norm_square), np.sqrt(norm_square) <= bandwidth

    return np.multiply(gaussian, bool_selector)
