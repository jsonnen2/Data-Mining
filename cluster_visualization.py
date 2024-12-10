
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import KBinsDiscretizer

'''
- histogram: classifier_acc / majority_class_acc
- Monie Graph
    Capitol M to represent a 1D amount of data
    Soft dash to follow the pattern of an M
    Use verticallity to split up distance between values 

Comp Graphics transformation
- vector of sorted data

xlimit(arr[x][0], arr[x][-1])
ylimit(0, 1)
x = arr
y = (x - x_min) / (x_max - x_min)
'''

# local_dir = "P5/_log3_knn"
# data = pd.read_csv(f"{local_dir}_summary_file.csv")
# point_estimate = data['classifier_mean'] / data['majority_class_mean']
# p_est = point_estimate.sort_values()
# xnew = p_est.to_numpy().reshape(-1,1)

data = pd.read_csv("data/train/data_about_review.csv")
point_estimate = data['hours_played']
p_est = point_estimate.sort_values()
xnew = p_est.to_numpy().reshape(-1,1)

orange = np.array([255, 219, 187]) / 255
deep_blue = np.array([70, 45, 160]) / 255
plt.figure(figsize=(13,5))
plt.xlabel("hours_played", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.title("Cluster Visualization", fontsize=18)


def monie_scatter_plot(data, cluster_edges, color):
    plt.xticks(cluster_edges, rotation=45)
    # cluster edges: list[int]
    n = len(cluster_edges)
    for i in range(0, n-1):
        ymin = cluster_edges[i]
        ymax = cluster_edges[i+1]
        x = data[(data > ymin) & (data < ymax)]
        y = (x.to_numpy() - ymin) / (ymax - ymin)
        if (i%2 == 1):
            y = 1 - y
        plt.scatter(x, y, marker='.', color=color)


def monie_line_plot(data, cluster_edges, linestyle, color):
    n = len(cluster_edges)
    for i in range(0, n-1):
        xmin = cluster_edges[i]
        xmax = cluster_edges[i+1]
        if (i%2 == 0):
            plt.plot([xmin, xmax], [0, 1], linestyle=linestyle, color=color)
        else:
            plt.plot([xmin, xmax], [1, 0], linestyle=linestyle, color=color)


# 2. find cluster edges
#   - index at these edges
# 1. Transform x and y. A corner at each cluster edge

alg = KBinsDiscretizer(n_bins=8, strategy='quantile')
alg.fit(xnew)
cluster_edges = alg.bin_edges_[0]
monie_line_plot(p_est, cluster_edges, '-', deep_blue)

alg = KBinsDiscretizer(n_bins=8, strategy='uniform')
alg.fit(np.log2(xnew+1))
cluster_edges = alg.bin_edges_[0]
cluster_edges = 2 ** cluster_edges - 1
monie_line_plot(p_est, cluster_edges, '--', "green")

# alg = KBinsDiscretizer(n_bins=8, strategy='kmeans')
# alg.fit(xnew)
# cluster_edges = alg.bin_edges_[0]
# monie_line_plot(p_est, cluster_edges, '-', deep_blue)

# alg = KBinsDiscretizer(n_bins=8, strategy='uniform')
# alg.fit(xnew)
# cluster_edges = alg.bin_edges_[0]
# monie_line_plot(p_est, cluster_edges, '--', "green")

plt.xlim(0, 200)
custom_legend = [
    Line2D([0], [0], color=deep_blue, linestyle='-', linewidth=2, label='Frequency'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='log'),

    # Line2D([0], [0], color=deep_blue, linestyle='-', linewidth=2, label='kmeans'),
    # Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='uniform'),
]


plt.legend(handles=custom_legend, loc='upper right')
plt.savefig("plots/hours_played_clustering/cluster_visualization_111")
plt.show()

