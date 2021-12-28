import csv

from mean_shift.mean_shift import MeanShift
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import time
import random


def generate_circular_shape_dataset():
    with open("datasets/data5.csv", "w", newline="") as csvfile:
        fieldnames = ['theta', 'r']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(300):
            writer.writerow({'theta': 2 * np.pi * random.random(), 'r': 8 + 2 * random.random()})
        for i in range(300):
            writer.writerow({'theta': 2 * np.pi * random.random(), 'r': 4 + 2 * random.random()})


def save_input(file_name):
    with open(file_name, "r") as f:
        input_string = f.read()
    return input_string.splitlines()[1:]


def convert_mean(mean):
    x, y = mean[0], mean[1]
    return [np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)]


def main():
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    file_name = "datasets/data1.csv"
    input_data = save_input(file_name)
    dataset = np.array([list(map(float, line.split(","))) for line in input_data])

    fig = plt.figure()

    start_time = time.time()
    mean_shifter = MeanShift(kernel="flat", kernel_parameters=[3], data_type="circular-linear")
    mean_shifter.fit(dataset)
    print("--- %s seconds ---" % (time.time() - start_time))
    centers = mean_shifter.cluster_centers_
    label = mean_shifter.labels_
    means = [convert_mean(mean_shifter.get_cluster_info(np.array(center))[0]) for center in centers]
    n_sample, _ = dataset.shape

    ax = fig.add_subplot(1, 2, 1)
    polar_ax = fig.add_subplot(1, 2, 2, projection="polar")
    ax.set_xlim([0, 2*np.pi])
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    for i in range(n_sample):
        ax.plot(dataset[i][0], dataset[i][1], "o", color=colors[label[i]], markersize=3)
    for i in range(n_sample):
        polar_ax.plot(dataset[i][0], dataset[i][1], "o", color=colors[label[i]], markersize=3)
    for mean in means:
        # ax.plot(mean[0], mean[1], "o", color="r", markersize=4)
        polar_ax.plot(mean[0], mean[1], "o", color="r", markersize=4)

    plt.show()


if __name__ == "__main__":
    main()
