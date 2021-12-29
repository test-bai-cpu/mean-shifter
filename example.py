from mean_shift.mean_shift import MeanShift
import numpy as np


def save_input(file_name):
    with open(file_name, "r") as f:
        input_string = f.read()
    return input_string.splitlines()[1:]


def main():
    # Read from input file and convert to numpy.ndarray
    file_name = "datasets/data1.csv"
    input_data = save_input(file_name)
    dataset = np.array([list(map(float, line.split(","))) for line in input_data])

    # To apply mean shift cluster to given dataset
    # Inputs are:
    # kernel: str: flat, gaussian, truncated_gaussian
    # kernel_parameters: list[float]:
    #                    For flat kernel: [bandwidth]
    #                    For Gaussian kernel: [beta]
    #                    For truncated Gaussian kernel: [bandwidth, beta]
    # data_type: str: circular-linear, linear
    mean_shifter = MeanShift(kernel="flat", kernel_parameters=[2], data_type="circular-linear")
    # mean_shifter = MeanShift(
    #     kernel="gaussian", kernel_parameters=[1], data_type="circular-linear"
    # )
    # mean_shifter = MeanShift(
    #     kernel="truncated_gaussian", kernel_parameters=[2,1], data_type="circular-linear"
    # )
    mean_shifter.fit(dataset)

    # To get cluster centers and labels for all samples in dataset
    cluster_centers, labels = mean_shifter.cluster_centers_, mean_shifter.labels_

    # To get cluster(mode) information
    cluster_info = mean_shifter.get_cluster_info(cluster_centers[0])

    # To predict the clusters of a dataset
    cluster_res = mean_shifter.predict(dataset)


if __name__ == "__main__":
    main()
