#!/usr/bin/env python
# Extension of SimSiam to RPR

import argparse
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

parser = argparse.ArgumentParser(description='Self supervised pre-training for RPRs')
parser.add_argument('--dataset_path',  help='path to dataset', default='/nfstemp/Datasets/7Scenes/')
parser.add_argument('--pairs_file', help='file with pairs', default='7scenes_training_pairs.csv')
parser.add_argument('--n_clusters', default=10, type=int, help='nuber of clusters)')


if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv(args.pairs_file)
    x1_ab, x2_ab, x3_a, q1_ab, q2_ab, q3_ab, q4_ab = df["x1_ab"].values, df["x2_ab"].values, df["x3_ab"].values, df["q1_ab"].values, df["q2_ab"].values, df["q3_ab"].values, df["q4_ab"].values
    X = []
    for i in range(len(x1_ab)):
        row = [x1_ab[i], x2_ab[i], x3_a[i], q1_ab[i], q2_ab[i], q3_ab[i], q4_ab[i]]
        X.append(row)
    #X = [x1_ab, x2_ab, x3_a, q1_ab, q2_ab, q3_ab, q4_ab]
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(X)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    np.savetxt("clusters.csv", kmeans.labels_, delimiter=",")








