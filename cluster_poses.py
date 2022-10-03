#!/usr/bin/env python
# Extension of SimSiam to RPR

import argparse
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from util.utils import pose_err
from torchvision import transforms
import torch
import kmeans1d

parser = argparse.ArgumentParser(description='Self supervised pre-training for RPRs')
parser.add_argument('--dataset_path',  help='path to dataset', default='/nfstemp/Datasets/7Scenes/')
parser.add_argument('--pairs_file', help='file with pairs', default='7scenes_training_pairs_fixed.csv')
parser.add_argument('--n_clusters', default=30, type=int, help='nuber of clusters)')

def cluster_ab(args):
    df = pd.read_csv(args.pairs_file)
    x1_ab, x2_ab, x3_ab, q1_ab, q2_ab, q3_ab, q4_ab = df["x1_ab"].values, df["x2_ab"].values, df["x3_ab"].values, df["q1_ab"].values, df["q2_ab"].values, df["q3_ab"].values, df["q4_ab"].values
    X = []
    for i in range(len(x1_ab)):
        row = [x1_ab[i], x2_ab[i], x3_ab[i], q1_ab[i], q2_ab[i], q3_ab[i], q4_ab[i]]
        X.append(row)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(X)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    np.savetxt("clusters_ab.csv", kmeans.labels_, delimiter=",")

def cluster_poses(args):
    df = pd.read_csv(args.pairs_file)
    x1_a, x2_a, x3_a, q1_a, q2_a, q3_a, q4_a = df["x1_a"].values, df["x2_a"].values, df["x3_a"].values, df["q1_a"].values, df["q2_a"].values, df["q3_a"].values, df["q4_a"].values
    x1_b, x2_b, x3_b, q1_b, q2_b, q3_b, q4_b = df["x1_b"].values, df["x2_b"].values, df["x3_b"].values, df["q1_b"].values, df["q2_b"].values, df["q3_b"].values, df["q4_b"].values
    #x1_ab, x2_ab, x3_ab, q1_ab, q2_ab, q3_ab, q4_ab = df["x1_ab"].values, df["x2_ab"].values, df["x3_ab"].values, df["q1_ab"].values, df["q2_ab"].values, df["q3_ab"].values, df["q4_ab"].values
    #img1, scene_a, scene_id_a, img2, scene_b, scene_id_b = df["img_path0"], df["scene_a"], df["scene_id_a"], df["img_path1"], df["scene_b"], df["scene_id_b"]
    X = []
    rows = []
    max_val = 100000
    table_length = len(x1_a)
    for i in range(table_length):
        pose_a = torch.from_numpy(np.array([x1_a[i], x2_a[i], x3_a[i], q1_a[i], q2_a[i], q3_a[i], q4_a[i]]))
        pose_b = torch.from_numpy(np.array([x1_b[i], x2_b[i], x3_b[i], q1_b[i], q2_b[i], q3_b[i], q4_b[i]]))
        posit_err, orient_err = pose_err(pose_a.unsqueeze(0), pose_b.unsqueeze(0))
        posit_err = posit_err[0].numpy()
        orient_err = orient_err[0].numpy()
        if np.isnan(posit_err) or np.isnan(orient_err):
            print('isnan idx: ' + str(i))
            posit_err = orient_err = max_val
            continue
        #rows.append([img1[i],scene_a[i],scene_id_a[i],x1_a[i], x2_a[i], x3_a[i], q1_a[i], q2_a[i], q3_a[i], q4_a[i],img2[i],scene_b[i],scene_id_b[i],x1_b[i], x2_b[i], x3_b[i], q1_b[i], q2_b[i], q3_b[i], q4_b[i],x1_ab[i], x2_ab[i], x3_ab[i], q1_ab[i], q2_ab[i], q3_ab[i], q4_ab[i] ])
        #X.append([posit_err, orient_err])
        X.append(orient_err[0][0])
    #kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(X)
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)
    clusters, centroids = kmeans1d.cluster(X, args.n_clusters)
    #print(clusters)
    print(centroids)
    #np.savetxt("clusters_poses.csv", kmeans.labels_, delimiter=",")
    np.savetxt("clusters_poses.csv", clusters, delimiter=",")
    #np.savetxt('7scenes_training_pairs_fixed.csv', rows, fmt='%s', delimiter=",")


def group_by_cluster(cluster_id, max_cluster):
    clusters = []
    cluster_id = np.asarray(cluster_id)
    for i in range(max_cluster):
        idx_in_cluster = np.where(cluster_id == i)
        clusters.append(np.array(idx_in_cluster))
    return clusters

def calc_mean_cluster_poses(args):
    df = pd.read_csv(args.pairs_file)
    x1_a, x2_a, x3_a, q1_a, q2_a, q3_a, q4_a = df["x1_a"].values, df["x2_a"].values, df["x3_a"].values, df["q1_a"].values, df["q2_a"].values, df["q3_a"].values, df["q4_a"].values
    x1_b, x2_b, x3_b, q1_b, q2_b, q3_b, q4_b = df["x1_b"].values, df["x2_b"].values, df["x3_b"].values, df["q1_b"].values, df["q2_b"].values, df["q3_b"].values, df["q4_b"].values
    cluster_id = df["cluster"]
    clusters = group_by_cluster(cluster_id, args.n_clusters)

    for idx in range(len(clusters)):
        list_pos_err = []
        list_ori_err = []
        for i in range(clusters[idx].size):
            pose_a = torch.from_numpy(np.array([x1_a[clusters[idx][0][i]], x2_a[clusters[idx][0][i]], x3_a[clusters[idx][0][i]], q1_a[clusters[idx][0][i]], q2_a[clusters[idx][0][i]], q3_a[clusters[idx][0][i]], q4_a[clusters[idx][0][i]]]))
            pose_b = torch.from_numpy(np.array([x1_b[clusters[idx][0][i]], x2_b[clusters[idx][0][i]], x3_b[clusters[idx][0][i]], q1_b[clusters[idx][0][i]], q2_b[clusters[idx][0][i]], q3_b[clusters[idx][0][i]], q4_b[clusters[idx][0][i]]]))
            posit_err, orient_err = pose_err(pose_a.unsqueeze(0), pose_b.unsqueeze(0))
            posit_err = posit_err[0].numpy()
            orient_err = orient_err[0].numpy()
            if np.isnan(posit_err) or np.isnan(orient_err):
                print('isnan idx: ' + str(i))
                continue
            list_pos_err.append(posit_err)
            list_ori_err.append(orient_err)
        mean_pos = np.mean(list_pos_err)
        mean_ori = np.mean(list_ori_err)
        std_pos = np.std(list_pos_err)
        std_ori = np.std(list_ori_err)
        print('cluster_id: ' + str(idx) + ' mean pos err: ' + str(mean_pos) + ' std pos err: ' + str(std_pos) + ' mean ori err: ' + str(mean_ori) + ' std ori err: ' + str(std_ori))

def cluster_mean(args):
    df = pd.read_csv('clusters.csv')
    x1_a = df["cluster"].values
    X = []
    table_length = len(x1_a)
    for i in range(table_length):
        X.append(x1_a[i])
        print(x1_a[i])
    clusters, centroids = kmeans1d.cluster(X, args.n_clusters)
    print(clusters)
    print(centroids)
    #np.savetxt("clusters_poses.csv", kmeans.labels_, delimiter=",")
    np.savetxt("clusters_poses.csv", clusters, delimiter=",")

    clusters_ = group_by_cluster(clusters, args.n_clusters)
    for idx in range(len(clusters_)):
        list1 = []
        for i in range(clusters_[idx].size):
            val = x1_a[clusters_[idx][0][i]]
            list1.append(val)
        mean_pos = np.mean(list1)
        print(np.mean(list1))

if __name__ == '__main__':
    args = parser.parse_args()

    #cluster_ab(args)
    #cluster_poses(args)
    calc_mean_cluster_poses(args)
    #cluster_mean(args)









