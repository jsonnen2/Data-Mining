import pandas as pd
import numpy as np


if __name__ == '__main__':
    # Two string builders
    # iterate across feature subsets

    feature_subsets = ["all", "chi-test", "textblob", "textblob-extra", "vote-count"]
    target_disc = ["range", "freq", "kmeans"]
    group_list = [3, 8, 12, 20]
    k_list = [5] # 87 is in a seperate file
    n = len(feature_subsets)
    table = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            # max table entry = 24
            for target in target_disc:
                for group in group_list:
                    for k in k_list:

                        string_builder_1 = f"F={feature_subsets[i]}_T={target}_GROUP={group}_K={k}"
                        string_builder_2 = f"F={feature_subsets[j]}_T={target}_GROUP={group}_K={k}"

                        t1 = 

                        table[i, j] += 
