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


def piss_extractor(list_important_point):
    def pis_extractor_from_pip(important_point_pos, list_important_point):
        start_pos = list_important_point[important_point_pos]
        end_pos = list_important_point[important_point_pos + 2]
        return np.asarray([start_pos, end_pos])

    return np.array([pis_extractor_from_pip(ip_pos, list_important_point)
                     for ip_pos in range(len(list_important_point) - 2)])


def pcs_extractor(pis, w, len_of_ts):
    if w == 0:
        return pis

    start_pos = pis[0] - w + 1
    start_pos = start_pos if start_pos >= 0 else 0
    end_pos = pis[1] + w
    end_pos = end_pos if end_pos < len_of_ts else len_of_ts

    return np.asarray([start_pos, end_pos])


def ci_extractor(time_series, list_important_point):
    def ci(time_series):
        return (time_series[:-1] - time_series[1:]) ** 2

    ts_ci = ci(time_series)
    list_ci_piss = []
    for j in range(len(list_important_point)-2):
        ci = np.sqrt(np.sum(ts_ci[list_important_point[j]:list_important_point[j+2] - 1]) + 0.001)
        list_ci_piss.append(ci)
    return ts_ci, list_ci_piss


def find_min_dist(pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list):
    sdist = calculate_subdist(matrix_t1, pis, list_start_pos, list_end_pos)

    len_pis = pis[1] - pis[0]
    len_pcs = pcs[1] - pcs[0]
    pcs_ci_temp = np.sum(pcs_ci_list[:len_pis-1])
    pcs_ci = np.sqrt(pcs_ci_temp + 0.001)
    d_ci = [(max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci))]
    for g in range(0, len_pcs - len_pis):
        pcs_ci_temp += pcs_ci_list[g + len_pis - 1]
        pcs_ci_temp -= pcs_ci_list[g]
        pcs_ci = np.sqrt(pcs_ci_temp + 0.001)
        d_ci.append((max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci)))
    d_ci = np.asarray(d_ci)

    return np.min(np.multiply(sdist, d_ci))


def PISD(ts_1, ts_1_piss, ts_1_ci, ts_1_ci_piss,
         ts_2, ts_2_piss, ts_2_ci, ts_2_ci_piss,
         list_start_pos, list_end_pos, w, min_dist):
    pisd = 0
    sum_len = calculate_sum_of_len(ts_1_piss, ts_2_piss)
    # Calculate matrix of t1 and t2
    matrix_t1, matrix_t2 = calculate_matrix(ts_1, ts_2, w)

    # Calculate distance from ts_1_piss to ts_1_piss
    for k in range(len(ts_1_piss)):
        pis = ts_1_piss[k]
        len_real_pis = pis[1] - pis[0]
        pcs = pcs_extractor(pis, w, len(ts_1))
        pis_ci = ts_1_ci_piss[k]
        pcs_ci_list = ts_2_ci[pcs[0]:pcs[1]-1]
        d_dist = find_min_dist(pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list)
        pisd += len_real_pis * d_dist
        if pisd / sum_len >= min_dist:
            break

    # Calculate distance from ts_2_piss to ts_2_piss
    for k in range(len(ts_2_piss)):
        pis = ts_2_piss[k]
        len_real_pis = pis[1] - pis[0]
        pcs = pcs_extractor(pis, w, len(ts_2))
        pis_ci = ts_2_ci_piss[k]
        pcs_ci_list = ts_1_ci[pcs[0]:pcs[1]-1]
        d_dist = find_min_dist(pis, pcs, matrix_t2, list_start_pos, list_end_pos, pis_ci, pcs_ci_list)
        pisd += len_real_pis * d_dist
        if pisd / sum_len >= min_dist:
            break
    # print(min_dist)
    return pisd / sum_len


def calculate_sum_of_len(train_ts_sq, test_ts_sq):
    sum_len = 0
    for k in range(len(train_ts_sq)):
        sum_len += train_ts_sq[k][1] - train_ts_sq[k][0]
    for k in range(len(test_ts_sq)):
        sum_len += test_ts_sq[k][1] - test_ts_sq[k][0]
    return sum_len


def calculate_matrix(ts_1, ts_2, w):
    matrix_1 = np.zeros((len(ts_1), 2 * w + 1))
    matrix_2 = np.zeros((len(ts_2), 2 * w + 1))

    list_dist = (ts_1 - ts_2) ** 2
    matrix_1[:, w] = list_dist
    matrix_2[:, w] = list_dist

    for i in range(w):
        list_dist = (ts_1[(i + 1):] - ts_2[:-(i + 1)]) ** 2
        matrix_1[(i + 1):, w - (i + 1)] = list_dist
        matrix_2[:-(i + 1), w + (i + 1)] = list_dist
        list_dist = (ts_2[(i + 1):] - ts_1[:-(i + 1)]) ** 2
        matrix_2[(i + 1):, w - (i + 1)] = list_dist
        matrix_1[:-(i + 1), w + (i + 1)] = list_dist

    return matrix_1, matrix_2


def calculate_subdist(matrix, pis, list_start_pos, list_end_pos):
    sub_matrix = matrix[pis[0]:pis[1], list_start_pos[pis[0]]:list_end_pos[pis[1]-1]]
    list_dist = np.sqrt(np.sum(sub_matrix, axis=0))
    return list_dist


class NN_PISD():
    def __init__(self, parameter):
        self.no_pip = parameter[0]
        self.w = parameter[1]
        self.len_of_ts = parameter[2]
        self.train_labels = None
        self.train_data = None
        self.train_data_pips = None
        self.train_data_piss = None
        self.train_data_ci = None
        self.train_data_ci_piss = None
        self.list_start_pos = None
        self.list_end_pos = None

    def fit(self, parameter):
        train_data, train_labels = parameter[0], parameter[1]
        self.len_of_ts = len(train_data[0])
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_data_pips = np.asarray([pips_extractor(t, self.no_pip) for t in train_data])
        self.train_data_piss = np.asarray([piss_extractor(self.train_data_pips[i])
                                          for i in range(len(train_data))])
        ci_return = [ci_extractor(self.train_data[i], self.train_data_pips[i])
                     for i in range(len(self.train_data))]
        self.train_data_ci = np.asarray([ci_return[i][0] for i in range(len(ci_return))])
        self.train_data_ci_piss = np.asarray([ci_return[i][1] for i in range(len(ci_return))])

    def sort_pips_piss_due_to_error_list(self, error_list):
        self.train_data = ucr.sort_data_by_error_list(self.train_data, error_list)
        self.train_labels = ucr.sort_data_by_error_list(self.train_labels, error_list)
        self.train_data, self.train_labels = np.asarray(self.train_data), np.asarray(self.train_labels)
        self.train_data_pips = ucr.sort_data_by_error_list(self.train_data_pips, error_list)
        self.train_data_piss = ucr.sort_data_by_error_list(self.train_data_piss, error_list)
        self.train_data_ci = ucr.sort_data_by_error_list(self.train_data_ci, error_list)
        self.train_data_ci_piss = ucr.sort_data_by_error_list(self.train_data_ci_piss, error_list)

    def set_w(self, w):
        self.w = w

        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int)*(self.w*2 + 1)
        for i in range(self.w):
            self.list_end_pos[-(i+1)] -= self.w - i
        for i in range(self.w-1):
            self.list_start_pos[i] += self.w - i - 1

    def set_no_pip(self, no_pip):
        self.no_pip = no_pip

    def predict(self, parameter):
        test_data, test_labels = parameter[0], parameter[1]
        incorrect_count = 0
        for i in range(len(test_data)):
            test_ts = test_data[i]
            test_ts_pips = pips_extractor(test_ts, self.no_pip)
            test_ts_piss = piss_extractor(test_ts_pips)
            test_ts_ci, test_ts_ci_piss = ci_extractor(test_ts, test_ts_pips)

            label = test_labels[i]
            min_dist = sys.float_info.max
            best_label = None
            for j in range(len(self.train_data)):
                train_ts = self.train_data[j]
                train_ts_piss = self.train_data_piss[j]
                train_ts_ci = self.train_data_ci[j]
                train_ts_ci_piss = self.train_data_ci_piss[j]

                distance = PISD(test_ts, test_ts_piss, test_ts_ci, test_ts_ci_piss,
                                train_ts, train_ts_piss, train_ts_ci, train_ts_ci_piss,
                                self.list_start_pos, self.list_end_pos, self.w, min_dist)
                if distance < min_dist:
                    min_dist = distance
                    best_label = self.train_labels[j]
            if best_label != label:
                incorrect_count += 1
            print("count: %s/%s - result: %s" % (i, len(test_data), incorrect_count), end="\r")
        return incorrect_count / len(test_data)

    def leave_one_out_tuning(self, parameter):
        best_accurate = parameter[0]

        train_data = self.train_data
        train_labels = self.train_labels
        train_data_piss = self.train_data_piss
        train_data_ci = self.train_data_ci
        train_data_ci_piss = self.train_data_ci_piss

        incorrect_count = 0
        error_list = []
        for i in range(len(train_data)):
            test_ts = train_data[i]
            test_ts_piss = train_data_piss[i]
            test_ts_ci = train_data_ci[i]
            test_ts_ci_piss = train_data_ci_piss[i]

            label = train_labels[i]
            min_dist = sys.float_info.max
            best_label = None
            for j in range(len(train_data)):
                if j != i:
                    train_ts = train_data[j]
                    train_ts_piss = train_data_piss[j]
                    train_ts_ci = train_data_ci[j]
                    train_ts_ci_piss = train_data_ci_piss[j]

                    distance = PISD(test_ts, test_ts_piss, test_ts_ci, test_ts_ci_piss,
                                    train_ts, train_ts_piss, train_ts_ci, train_ts_ci_piss,
                                    self.list_start_pos, self.list_end_pos, self.w, min_dist)
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





# t1 = np.asarray([1,2,3,4,5,6,7,8,9])
# t2 = np.asarray([1,5,3,2,7,5,6,3,1])
# len_of_ts = len(t1)
# w = 2
# m1, m2 = calculate_matrix(t1,t2,w)
# list_start_pos = np.ones(len_of_ts, dtype=int)
# list_end_pos = np.ones(len_of_ts, dtype=int) * (w * 2 + 1)
# for i in range(w):
#     list_end_pos[-(i + 1)] -= w - i
# for i in range(w - 1):
#     list_start_pos[i] += w - i - 1
#
# pis = np.asarray([0,8])
# s = calculate_subdist(m1, pis, list_start_pos, list_end_pos)
