import sys
import numpy as np
import scipy.stats as stats
import ucr_data_loader as ucr


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
        min_dist = min(dist, min_dist)

    return min_dist


def calculate_sum_of_len(train_ts_sq, test_ts_sq):
    sum_len = 0
    for sq in train_ts_sq:
        sum_len += len(sq)
    for sq in test_ts_sq:
        sum_len += len(sq)

    return sum_len


def PISD(ts_1, ts_1_pips, ts_1_piss, ts_2, ts_2_pips, ts_2_piss, w, min_dist):
    distance = 0
    # Calculate distance from ts_1
    sum_len = calculate_sum_of_len(ts_1_piss, ts_2_piss)
    for k in range(len(ts_1_pips) - 2):
        pis = ts_1_piss[k]
        pcs = pcs_extractor(ts_2, k, ts_1_pips, w)
        distance += len(pis) * subdist(pis, pcs)
        if distance / sum_len >= min_dist:
            break
    # Calculate distance from ts_2
    for k in range(len(ts_2_pips) - 2):
        pis = ts_2_piss[k]
        pcs = pcs_extractor(ts_1, k, ts_2_pips, w)
        distance += len(pis) * subdist(pis, pcs)
        if distance / sum_len >= min_dist:
            break

    return distance / sum_len


class NN_PISD():
    def __init__(self, parameter):
        self.no_pip = parameter[0]
        self.w = parameter[1]
        self.train_labels = None
        self.train_data = None
        self.train_data_pips = None
        self.train_data_piss = None

    def fit(self, parameter):
        train_data, train_labels = parameter[0], parameter[1]
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_data_pips = [pips_extractor(t, self.no_pip) for t in train_data]
        self.train_data_piss = [piss_extractor(self.train_data[i], self.train_data_pips[i])
                                for i in range(len(train_data))]

    def sort_pips_piss_due_to_error_list(self, error_list):
        self.train_data =  ucr.sort_data_by_error_list(self.train_data, error_list)
        self.train_labels = ucr.sort_data_by_error_list(self.train_labels, error_list)
        self.train_data, self.train_labels = np.asarray(self.train_data), np.asarray(self.train_labels)
        self.train_data_pips = ucr.sort_data_by_error_list(self.train_data_pips, error_list)
        self.train_data_piss = ucr.sort_data_by_error_list(self.train_data_piss, error_list)

    def set_w(self, w):
        self.w = w

    def set_no_pip(self, no_pip):
        self.no_pip = no_pip

    def predict(self, parameter):
        test_data, test_labels = parameter[0], parameter[1]
        incorrect_count = 0
        for i in range(len(test_data)):
            test_ts = test_data[i]
            test_ts_pips = pips_extractor(test_ts, self.no_pip)
            test_ts_piss = piss_extractor(test_ts, test_ts_pips)

            label = test_labels[i]
            min_dist = np.inf
            best_label = None
            for j in range(len(self.train_data)):
                train_ts = self.train_data[j]
                train_ts_pips = self.train_data_pips[j]
                train_ts_piss = self.train_data_piss[j]
                distance = PISD(test_ts, test_ts_pips, test_ts_piss,
                                train_ts, train_ts_pips, train_ts_piss,
                                self.w, min_dist)
                if distance < min_dist:
                    min_dist = distance
                    best_label = self.train_labels[j]
            if best_label != label:
                incorrect_count += 1
            print("count: %s/%s - result: %s" % (i,len(test_data), incorrect_count), end="\r")

        return incorrect_count / len(test_data)

    def leave_one_out_tuning(self, parameter):
        best_accurate = parameter[0]
        train_data = self.train_data
        train_labels = self.train_labels
        train_data_pips = self.train_data_pips
        train_data_piss = self.train_data_piss

        incorrect_count = 0
        error_list = []
        for i in range(len(train_data)):
            test_ts = train_data[i]
            test_ts_pips = train_data_pips[i]
            test_ts_piss = train_data_piss[i]
            label = train_labels[i]
            min_dist = sys.float_info.max
            best_label = None
            for j in range(len(train_data)):
                if j != i:
                    train_ts = train_data[j]
                    train_ts_pips = train_data_pips[j]
                    train_ts_piss = train_data_piss[j]
                    distance = PISD(test_ts, test_ts_pips, test_ts_piss,
                                    train_ts, train_ts_pips, train_ts_piss,
                                    self.w, min_dist)
                    if distance < min_dist:
                        min_dist = distance
                        best_label = train_labels[j]
            if best_label != label:
                incorrect_count += 1
                error_list.append(False)
            else:
                error_list.append(True)
            if round(incorrect_count / len(train_data), 3) >= best_accurate:
                return round(incorrect_count / len(train_data) + 10, 3), error_list
            print("count: %s/%s - result: %s" % (i, len(train_data), incorrect_count), end="\r")

        return round(incorrect_count / len(train_data), 3), error_list