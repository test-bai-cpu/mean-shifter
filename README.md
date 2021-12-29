# Mean-shift-cluster

A tool, mean-shift-cluster is developed, which accepts both circular-linear data and linear data.
Mean-shift-cluster supports three kernels: `flat kernel`, `Gaussian kernel` and `truncated 
Gaussian kernel`.


## Usage
Mean-shift-cluster provides three APIs:
- `fit`: Fit input dataset using mean shift algorithm. Cluster centers and labels
for each sample can be accessed.
- `get_cluster_info`: Provide basic information of each cluster(mode), using input dataset.
- `predict`: Predict the cluster results of any dataset, using the fitted mean-shift-cluster instance.

The inputs are:
- `kernel`: Type is string. Options are: flat, gaussian, truncated_gaussian
- `kernel_parameters`: type is List[float].  
  - For flat kernel: [bandwidth]
  - For Gaussian kernel: [beta]
  - For truncated Gaussian kernel: [bandwidth, beta]
- `data_type`: Type is string. Options are: circular-linear, linear

A simple usage example is in [example.py](https://github.com/test-bai-cpu/mean-shifter/blob/master/example.py). 
And more details are in the report.


