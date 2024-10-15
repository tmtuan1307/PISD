import numpy as np
import sys
import copy
import sklearn.metrics.cluster as clus
import random
import operator
import scipy.stats as stats
from scipy.special import comb


def pips_extractor(time_series, num_of_ip):
    def pd_distance(parameter):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = parameter[0], parameter[1], parameter[2], \
                                             parameter[3], parameter[4], parameter[5]
        b = -1
        a = (p2_y - p3_y) / (p2_x - p3_x)
        c = p2_y - p2_x * a
        return abs(b * p1_y + a * p1_x + c) / np.sqrt(a ** 2 + b ** 2)

    # Init list of index of time_series with z_norm
    list_index_z = stats.zscore(list(range(len(time_series))))

    # Add first and last element as the two first element in pips
    # Remember that the pips just store the position of the point
    pips = [0, (len(time_series) - 1)]
    remain_pos = list(range(1, (len(time_series) - 1)))

    for i in range(num_of_ip - 2):
        biggest_dist = -1
        biggest_dist_pos = -1
        for j in remain_pos:
            p_y = time_series[j]
            p_x = list_index_z[j]
            # Find the adjective point (p2,p3) of p in pips
            p2_x = p2_y = p3_x = p3_y = None
            for g in range(len(pips)):
                end_pos = pips[g]
                if j < end_pos:
                    p2_y, p2_x = time_series[end_pos], list_index_z[end_pos]
                    start_pos = pips[g - 1]
                    p3_y, p3_x = time_series[start_pos], list_index_z[start_pos]
                    break
            # Calculate the distance of p to p2,p3
            distance = pd_distance(parameter=[p_x, p_y, p2_x, p2_y, p3_x, p3_y])
            if distance > biggest_dist:
                biggest_dist, biggest_dist_pos = distance, j
        # Add biggest_dist_pos into pips
        if biggest_dist_pos == -1:
            print("-dist errpr-")
        pips.append(biggest_dist_pos)
        pips.sort()
        # Remove biggest_dist_pos from remain_pos
        remain_pos.remove(biggest_dist_pos)

    return pips


def piss_extractor(time_series, list_important_point):
    def pis_extractor_from_pip(time_series, important_point_pos, list_important_point):
        start_pos = list_important_point[important_point_pos]
        end_pos = list_important_point[important_point_pos + 2]
        return time_series[start_pos:end_pos]

    return np.array([pis_extractor_from_pip(time_series, ip_pos, list_important_point) for ip_pos in
                     range(len(list_important_point) - 2)])


def pcs_extractor(time_series, important_point_pos, list_important_point, w):
    if w == 0:
        start_pos = list_important_point[important_point_pos]
        end_pos = list_important_point[important_point_pos + 2]
        return time_series[start_pos:end_pos]

    start_pos = list_important_point[important_point_pos] - w + 1
    start_pos = start_pos if start_pos >= 0 else 0
    end_pos = list_important_point[important_point_pos + 2] + w
    end_pos = end_pos if end_pos < len(time_series) else len(time_series)

    return time_series[start_pos:end_pos]


def subdist(t1,t2):
    len_t1, len_t2 = len(t1), len(t2)
    diff_len = len_t2 - len_t1 + 1
    min_dist = np.inf
    min_pos = -1
    ci_t1 = np.sqrt(np.sum((t1[:-1] - t1[1:]) ** 2) + 0.001)
    temp_t2 = (t2[:-1] - t2[1:]) ** 2
    first_sum = np.sum(temp_t2[:len_t1 - 1])
    for i in range(diff_len):
        diff_ts = np.sqrt(np.sum((t1 - t2[i:i + len_t1]) ** 2))
        ci_sq_t2 = np.sqrt(first_sum + 0.001)
        if i < diff_len - 1:
            first_sum += temp_t2[i + len_t1 - 1]
            first_sum -= temp_t2[i]

        dist = diff_ts * (max(ci_t1, ci_sq_t2) / min(ci_t1, ci_sq_t2))
        if dist < min_dist:
            min_dist = dist
            min_pos = i

    return min_dist, min_pos


def calculate_sum_of_len(ts_1_piss, ts_2_piss):
    sum_len = 0
    for pis in ts_1_piss:
        sum_len += len(pis)
    for pis in ts_2_piss:
        sum_len += len(pis)
    return sum_len


def PISD(ts_1, ts_1_pips, ts_1_piss, ts_2, ts_2_pips, ts_2_piss, w):
    distance = 0
    # Calculate distance from train_piss to train_pcss
    sum_len = calculate_sum_of_len(ts_1_piss, ts_2_piss)
    for k in range(len(ts_1_pips) - 2):
        pis = ts_1_piss[k]
        pcs = pcs_extractor(ts_2, k, ts_1_pips, w)
        sdist, min_pos = subdist(pis, pcs)
        distance += len(pis) * sdist

    # Calculate distance from test_piss to test_pcss
    list_averaging_ts = [[] for i in range(len(ts_2_piss)+2)]
    list_sum_weighted = [0 for i in range(len(ts_2_piss)+2)]
    for k in range(len(ts_2_pips) - 2):
        pis = ts_2_piss[k]
        pcs = pcs_extractor(ts_1, k, ts_2_pips, w)
        sdist, min_pos = subdist(pis, pcs)

        weight = 1 / (sdist + 0.1)
        break_point = ts_2_pips[k+1] - ts_2_pips[k]
        selected_pcs = pcs[min_pos:min_pos+len(pis)]
        list_averaging_ts[k].append(selected_pcs[:break_point]*weight)
        list_averaging_ts[k+1].append(selected_pcs[break_point:]*weight)
        list_sum_weighted[k] += weight
        list_sum_weighted[k+1] += weight

        distance += len(pis) * sdist
    distance /= sum_len

    total_weight = 1/(distance + 0.1)
    list_averaging_ts[-1].append(np.asarray([ts_1[-1]*total_weight]))
    list_sum_weighted[-1] += total_weight

    return distance, list_averaging_ts, list_sum_weighted


def rand_index_score(clusters, classes):
    classes = np.asarray(classes)
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def find_cluster(data, data_pips, data_piss, centroids, centroids_pips, centroids_piss, w):
    cluster_list = []
    dist_ts_centroid = []
    avg_ts_list = []
    sum_weight_list = []
    for i in range(len(data)):
        ts, ts_pips, ts_piss = data[i], data_pips[i], data_piss[i]
        min_dist, min_label, label = np.inf, -1, 1
        min_avg_ts, min_sum_weight = [], []
        for j in range(len(centroids)):
            ct, ct_pips, ct_piss = centroids[j], centroids_pips[j], centroids_piss[j]
            dist, list_averaging_ts, list_sum_weighted = PISD(ts, ts_pips, ts_piss, ct, ct_pips, ct_piss, w)
            if dist < min_dist:
                min_dist, min_label = dist, label
                min_avg_ts, min_sum_weight = list_averaging_ts, list_sum_weighted
            label += 1
        cluster_list.append(min_label)
        dist_ts_centroid.append(min_dist)
        avg_ts_list.append(min_avg_ts)
        sum_weight_list.append(min_sum_weight)
    return np.asarray(cluster_list), np.asarray(dist_ts_centroid), np.asarray(avg_ts_list), np.asarray(sum_weight_list)


def pisd_averaging(avg_ts_list, avg_sum_weight_list):
    data_from_each_piss = [[] for i in range(len(avg_ts_list[0]))]
    for i in range(len(avg_ts_list)):
        element = avg_ts_list[i]
        for j in range(len(element)):
            data_from_each_piss[j].append(np.sum(element[j],axis=0))
    sum_weight = np.sum(avg_sum_weight_list,axis=0)
    averaging_ts = [np.sum(data_from_each_piss[i],axis=0)/sum_weight[i] for i in range(len(data_from_each_piss))]
    averaging_ts = np.concatenate(averaging_ts)

    return averaging_ts


def update_centroids(data, cluster_list, centroids, avg_ts_list, sum_weight_list):
    empty_clusters = []
    new_centroids = []
    for i in range(1, len(centroids) + 1):
        cen_data = data[np.where(cluster_list == i)]
        cen_avg_ts_list = avg_ts_list[np.where(cluster_list == i)]
        cen_sum_weight_list = sum_weight_list[np.where(cluster_list == i)]
        if len(cen_data) == 0:
            # add the cluster index to the empty clusters list
            empty_clusters.append(i - 1)
            new_centroids.append([])
            # skip averaging an empty cluster
            continue

        new_centroid = pisd_averaging(cen_avg_ts_list, cen_sum_weight_list)
        new_centroids.append(new_centroid)

    return np.asarray(new_centroids), empty_clusters


def find_init_centroid(data, data_pips, data_piss, no_cluster):
    centroids = []
    centroids_pips = []
    centroids_piss = []
    list_current = []
    for g in range(no_cluster):
        r = random.randint(0, len(data) - 1)
        while list_current.__contains__(r):
            r = random.randint(0, len(data) - 1)
        centroids.append(data[r])
        centroids_pips.append(data_pips[r])
        centroids_piss.append(data_piss[r])
        list_current.append(r)
    return centroids, centroids_pips, centroids_piss


def find_init_centroid_plus(data, data_pips, data_piss, no_cluster, w, test_labels):
    centroids = [data[0]]
    centroids_pips = [data_pips[0]]
    centroids_piss = [data_piss[0]]
    selected_pos = [0]
    for g in range(no_cluster-1):
        max_dist = -1
        max_pos = -1
        for i in range(len(data)):
            if not selected_pos.__contains__(i):
                sum_dist = 0
                for c in range(len(centroids)):
                    sum_dist += PISD(data[i], data_pips[i], data_piss[i],
                                     centroids[c], centroids_pips[c], centroids_piss[c], w)[0]
                if sum_dist > max_dist:
                    max_dist = sum_dist
                    max_pos = i

        centroids.append(data[max_pos])
        centroids_pips.append(data_pips[max_pos])
        centroids_piss.append(data_piss[max_pos])
        selected_pos.append(max_pos)

    labels = [test_labels[pos] for pos in selected_pos]
    print(labels)
    return centroids, centroids_pips, centroids_piss


class KMEAN_PISD():
    def __init__(self, parameter):
        self.no_of_cluster = parameter[0]
        self.no_pip = parameter[1]
        self.w = parameter[2]
        self.maximum_loop = parameter[3]
        self.train_data = None
        self.train_labels = None
        self.train_data_pips = None
        self.train_data_piss = None

    def set_w(self, w):
        self.w = w

    def set_no_pip(self, no_pip):
        self.no_pip = no_pip

    def fit(self, parameter):
        self.train_data = parameter[0]
        train_labels_str = parameter[1]
        self.train_labels = [int(l) + 2 for l in train_labels_str]
        self.train_data_pips = [pips_extractor(t, self.no_pip) for t in self.train_data]
        self.train_data_piss = [piss_extractor(self.train_data[i], self.train_data_pips[i])
                                for i in range(len(self.train_data))]

    def cluster(self, parameter):
        test_data = parameter[0]
        test_labels_str = parameter[1]
        test_labels = [int(l) + 2 for l in test_labels_str]
        test_data_pips = [pips_extractor(t, self.no_pip) for t in test_data]
        test_data_piss = [piss_extractor(test_data[i], test_data_pips[i])
                          for i in range(len(test_data))]

        # Get central as first k node
        centroids, centroids_pips, centroids_piss = \
            find_init_centroid(test_data, test_data_pips, test_data_piss, self.no_of_cluster)

        count = 0
        break_count = 0
        cluster_list = []
        while count < self.maximum_loop:
            # find the nearest centroid for each ts and assign it to its cluster
            cluster_list, dist_ts_centroid, avg_ts_list, sum_weight_list = find_cluster \
                (test_data, test_data_pips, test_data_piss, centroids, centroids_pips, centroids_piss, self.w)

            # update centroid of each cluster
            old_centroids = copy.deepcopy(centroids)
            old_cluster_list = copy.deepcopy(cluster_list)

            centroids, empty_clusters = update_centroids(test_data, cluster_list, centroids,
                                                         avg_ts_list, sum_weight_list)
            if len(empty_clusters) > 0:
                # get l farest ts from its centroids
                top_l_farest_ts = np.argpartition(dist_ts_centroid, -len(empty_clusters))[-len(empty_clusters):]
                # loop through the empty clusters
                for i, idx_clust in enumerate(empty_clusters):
                    # replace the empty cluster with the farest time series from its old cluster
                    centroids[idx_clust] = test_data[top_l_farest_ts[i]]

            centroids_pips = [pips_extractor(t, self.no_pip) for t in centroids]
            centroids_piss = [piss_extractor(centroids[i], centroids_pips[i])
                              for i in range(len(centroids))]

            if np.array_equal(cluster_list, old_cluster_list):
                break_count += 1
                if np.array_equal(centroids, old_centroids) or break_count == 20:
                    break
            else:
                break_count = 0
            count += 1

        return rand_index_score(cluster_list, test_labels), \
               clus.adjusted_rand_score(cluster_list, test_labels), \
               clus.normalized_mutual_info_score(cluster_list, test_labels)

    def leave_one_out_tuning(self):
        train_data = self.train_data
        train_labels = self.train_labels
        train_data_pips = self.train_data_pips
        train_data_piss = self.train_data_piss
        # Get central as first k node
        centroids, centroids_pips, centroids_piss = \
            find_init_centroid(train_data, train_data_pips, train_data_piss, self.no_of_cluster)

        count = 0
        break_count = 0
        cluster_list = []
        while count < self.maximum_loop:
            # find the nearest centroid for each ts and assign it to its cluster
            cluster_list, dist_ts_centroid, avg_ts_list, sum_weight_list = find_cluster \
                (train_data, train_data_pips, train_data_piss, centroids, centroids_pips, centroids_piss, self.w)
            # update centroid of each cluster
            old_centroids = copy.deepcopy(centroids)
            old_cluster_list = copy.deepcopy(cluster_list)

            centroids, empty_clusters = update_centroids(train_data, cluster_list, centroids,
                                                         avg_ts_list, sum_weight_list)
            if len(empty_clusters) > 0:
                # get l farest ts from its centroids
                top_l_farest_ts = np.argpartition(dist_ts_centroid, -len(empty_clusters))[-len(empty_clusters):]
                # loop through the empty clusters
                for i, idx_clust in enumerate(empty_clusters):
                    # replace the empty cluster with the farest time series from its old cluster
                    centroids[idx_clust] = train_data[top_l_farest_ts[i]]

            centroids_pips = [pips_extractor(t, self.no_pip) for t in centroids]
            centroids_piss = [piss_extractor(centroids[i], centroids_pips[i])
                              for i in range(len(centroids))]

            if np.array_equal(cluster_list, old_cluster_list):
                break_count += 1
                if np.array_equal(centroids, old_centroids) or break_count == 20:
                    break
            else:
                break_count = 0
            count += 1

        return rand_index_score(cluster_list, train_labels)
