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
    mean_shifter = MeanShift(kernel="flat", kernel_parameters=[3], data_type="circular-linear")
    mean_shifter.fit(dataset)

    # To get cluster centers and labels for all samples in dataset
    cluster_centers, labels = mean_shifter.cluster_centers_, mean_shifter.labels_

    # To get cluster(mode) information
    cluster_info = mean_shifter.get_cluster_info(cluster_centers[0])

    # To predict the clusters of a dataset
    cluster_res = mean_shifter.predict(dataset)


if __name__ == "__main__":
    main()
