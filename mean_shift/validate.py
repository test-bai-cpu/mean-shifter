from typing import List, Type

import numpy as np
from mean_shift.exceptions import NotFittedError, KernelInputError


def validate_data(dataset: np.ndarray, data_type: str) -> np.ndarray:
    # Validate input circular-linear data and convert from polar to cartesian coordinates.
    #
    # Parameters
    # ----------
    # dataset : array-like of shape (n_samples, n_features)
    #     The input samples.
    #
    # Returns
    # -------
    # out : (n_samples, n_features)
    #     The validated input. The output data is in cartesian coordinates.
    if not isinstance(dataset, np.ndarray) or dataset.shape[1] != 2:
        raise TypeError(
            "Could not find input dataset or dataset type is not correct. "
            "Input dataset type should be numpy.ndarray, and with two features: theta, r"
        )

    if data_type == "linear":
        return dataset

    if any(theta <= 0 or theta >= 2 * np.pi for theta in dataset[:, 0]):
        raise ValueError(
            "In circular-linear dataset, theta value should be within range [0, 2*pi]"
        )

    return np.array(
        [[sample[1] * np.cos(sample[0]), sample[1] * np.sin(sample[0])] for sample in dataset]
    )


def validate_fitted_instance(mean_shift_instance) -> None:
    if (mean_shift_instance.cluster_centers_ is None) or (mean_shift_instance.labels_ is None):
        raise NotFittedError(
            "This instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using mean_shift tool."
        )


def validate_kernel_inputs(kernel: str, kernel_parameters: List) -> None:
    if kernel_parameters is None or not isinstance(kernel_parameters, list):
        raise KernelInputError(
            "Could not find Kernel parameter, or input format is wrong. "
            "Examples: "
            "kernel='flat', kernel_parameters=[1]"
            "kernel='gaussian', kernel_parameters=[0.1]"
            "kernel='truncated_gaussian', kernel_parameters=[1, 0.1]"
        )

    if kernel in ["flat", "gaussian"] and (
            not len(kernel_parameters) == 1 or
            kernel_parameters[0] <= 0):
        raise KernelInputError(
            "For flat or Gaussian kernel, parameter needs to be float and greater than zero."
        )

    if kernel == "truncated_gaussian" and (
            not len(kernel_parameters) == 2 or
            any(parameter <= 0 for parameter in kernel_parameters)
    ):
        raise KernelInputError(
            "Given kernel parameter is wrong. For truncated Gaussian kernel, "
            "two parameters are required: beta and bandwidth."
            "And parameters needs to be greater than zero, "
        )

    if kernel not in ["flat", "gaussian", "truncated_gaussian"]:
        raise KernelInputError(
            "Given kernel name is not supported. "
            "Available options are: flat, gaussian and truncated_gaussian"
        )
