import math
from typing import List, Tuple, Any

import numpy as np

from joblib import Parallel  # type: ignore
from sklearn.utils.fixes import delayed  # type: ignore

from mean_shift.validate import validate_fitted_instance, validate_data, validate_kernel_inputs
from mean_shift import utils


class MeanShift:
    dataset_: np.ndarray
    cluster_centers_: np.ndarray
    labels_: np.ndarray
    n_iter_: int

    def __init__(
            self,
            *,
            kernel: str = "flat",
            kernel_parameters: List[float] = None,
            max_iter: int = 300,
            data_type: str = "circular_linear",
            cluster_all: bool = True,
    ) -> None:
        self.kernel = kernel
        self.kernel_parameters = kernel_parameters if kernel_parameters else []
        self.max_iter = max_iter
        self.data_type = data_type
        self.cluster_all = cluster_all

    def fit(self, dataset: np.ndarray) -> None:
        # Perform clustering.
        #
        # Parameters
        # ----------
        # dataset : array-like of shape (n_samples, n_features)
        #     Samples to cluster.
        #
        # Returns
        # -------
        # None
        validate_kernel_inputs(self.kernel, self.kernel_parameters)
        self.dataset_ = validate_data(dataset, self.data_type)

        # Execute iterations on all seeds.
        seeds = self._generate_seeds()

        # all_res = [self._mean_shift_single_seed(seed) for seed in seeds]
        all_res = Parallel()(delayed(self._mean_shift_single_seed)(seed) for seed in seeds)

        # copy results in a dictionary
        center_intensity_dict = {}
        for i in range(len(seeds)):
            if all_res[i][1]:  # i.e. len(points_within) > 0
                center_intensity_dict[all_res[i][0]] = all_res[i][1]

        self.n_iter_ = max([x[2] for x in all_res])

        if not center_intensity_dict:
            # nothing near seeds
            raise ValueError(
                "No point was within bandwidth=%f of any seed. Try a different seeding"
                " strategy or increase the bandwidth." % self.kernel_parameters[0]
            )

        # POST PROCESSING: remove near duplicate points
        # If the distance between two kernels is less than the bandwidth,
        # then we have to remove one because it is a duplicate. Remove the
        # one with fewer points.
        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        cluster_centers_list = [sorted_centers[0]]

        if self.kernel in ["flat", "truncated_gaussian"]:
            bandwidth = self.kernel_parameters[0]
        else:
            bandwidth = math.sqrt(1 / (2 * self.kernel_parameters[0]))
        for sorted_center in sorted_centers[1:]:
            if any(utils.get_euclidean_distance(x, sorted_center) < bandwidth
                   for x in cluster_centers_list):
                continue
            cluster_centers_list.append(sorted_center)
        cluster_centers = np.round(np.array(cluster_centers_list), 2)

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        n_samples = dataset.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        distances, indices = self._get_nearest_cluster_center_and_distance(cluster_centers)
        if self.cluster_all:
            labels = indices.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= self.kernel_parameters[0]
            labels[bool_selector] = indices.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        # Predict the closest cluster each sample in X belongs to.
        #
        # Parameters
        # ----------
        # dataset : array-like of shape (n_samples, n_features)
        #     New data to predict.
        #
        # Returns
        # -------
        # labels : ndarray of shape (n_samples,)
        #     Index of the cluster each sample belongs to.
        validate_fitted_instance(self)
        dataset = validate_data(dataset, self.data_type)
        indices = np.array(
            [np.argmin([utils.get_euclidean_distance(sample, cluster_center)  # type: ignore
                        for cluster_center in self.cluster_centers_]) for sample in dataset])

        return indices

    def get_cluster_info(
            self, cluster_center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # In the given dataset, get information of one cluster
        #
        # Parameters
        # ----------
        # cluster_center : ndarray of shape
        #     Position of cluster center
        #
        # Returns
        # -------
        # cluster_info : tuple
        #     (mean, variance, covariance)
        validate_fitted_instance(self)
        index = np.where(np.all(self.cluster_centers_ == cluster_center, axis=1))[0]
        if not len(index):
            raise ValueError("The given cluster center does not exist.")
        data = self.dataset_[self.labels_ == index]

        mean = np.mean(data, axis=0)
        variance = np.var(data, axis=0)
        cov = np.cov(data)

        return mean, variance, cov

    def _generate_seeds(self) -> np.ndarray:
        seeds = self.dataset_
        return seeds

    def _mean_shift_single_seed(self, seed: np.ndarray) -> Tuple[Any, int, int]:
        # A separate function for each seed's iterative loop
        # For each seed, climb gradient until convergence or max_iter
        # convergence_threshold = 1e-3 * self.kernel_parameters[0]  # when mean has converged
        convergence_threshold = 1e-6
        # params initialization
        completed_iterations = 0
        current_mean = seed
        mean_shift = np.inf

        while mean_shift > convergence_threshold and completed_iterations < self.max_iter:
            previous_mean = current_mean
            current_mean = self._get_mean_value(current_mean)
            mean_shift = np.linalg.norm(current_mean - previous_mean)
            completed_iterations += 1

        neighbor_count = self._get_nearest_neighbor_by_radius(current_mean)

        return tuple(current_mean), neighbor_count, completed_iterations

    def _get_mean_value(self, cluster_center: np.ndarray) -> np.ndarray:
        kernel_res = self._apply_kernel(cluster_center)
        mean = np.dot(kernel_res, self.dataset_) / sum(kernel_res)

        return mean

    def _apply_kernel(self, cluster_center: np.ndarray) -> np.ndarray:
        # Get weighted points based on kernel options
        if self.kernel == "flat":
            return utils.get_flat_kernel(cluster_center, self.dataset_, self.kernel_parameters)
        elif self.kernel == "gaussian":
            return utils.get_gaussian_kernel(cluster_center, self.dataset_, self.kernel_parameters)
        else:  # self.kernel == "truncated_gaussian":
            return utils.get_truncated_gaussian_kernel(
                cluster_center, self.dataset_, self.kernel_parameters
            )

    def _get_nearest_neighbor_by_radius(self, cluster_center: np.ndarray) -> int:
        # Return numbers of neighbors near cluster center.
        # For flat and truncated Gaussian kernels, bandwidth(lambda) is used as radius
        # For Gaussian kernel, sigma = sqrt(1/2*beta), is used as radius
        if self.kernel in ["flat", "truncated_gaussian"]:
            bandwidth = self.kernel_parameters[0]
        else:
            bandwidth = math.sqrt(1 / (2 * self.kernel_parameters[0]))
        neighbor_count = sum(map(lambda x:
                                 utils.get_euclidean_distance(x, cluster_center) <
                                 bandwidth,
                                 self.dataset_)
                             )

        return int(neighbor_count)

    def _get_nearest_cluster_center_and_distance(
            self, cluster_centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Assign samples to the nearest cluster
        distances, indices = [], []

        for sample in self.dataset_:
            distances = [
                utils.get_euclidean_distance(sample, cluster_center)
                for cluster_center in cluster_centers
            ]
            min_index = np.argmin(distances)
            distances.append(distances[min_index])
            indices.append(min_index)

        return np.array(distances), np.array(indices)
