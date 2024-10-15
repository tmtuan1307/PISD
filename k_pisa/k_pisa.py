import numpy as np
import argparse
import config as config
import copy
import ucr_data_loader as data_loader
from k_pisa_model import KMEAN_PISD
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='ArrowHead', type=str, help="You can use dataset_name directly, please "
                                                                             "see in the config.py for all dataset name")
    parser.add_argument("--dataset_pos", default=0, type=int, help="Or You can just use dataset pos in the"
                                                                            "config.py file")
    parser.add_argument("--first_w", default=18, type=int, help="The first window size for tuning num_of_pip parameter")
    parser.add_argument("--max_percent", default=20, type=int, help="Maximum percent of window size")
    parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterator for looping k-mean")
    args = parser.parse_args()

    all_dataset_name = config.CLUSTER_DATASET_NAME
    cluster_flag = "PISA"

    if args.dataset_name != "":
        name = args.dataset_name
    else:
        name = all_dataset_name[args.dataset_pos]
    print("Algorithm: %s - dataset: %s" % (cluster_flag,name))

    train_data, train_labels, test_data, test_labels = data_loader.load_dataset_zscore(name, config.DATASET_PATH)
    num_of_classes = len(np.unique(train_labels))
    store_train_data = copy.deepcopy(train_data)
    store_train_labels = copy.deepcopy(train_labels)
    len_of_ts = len(train_data[0])

    # num_of_pip parameter
    pip_count = (len_of_ts / 10 - 3) / 10
    if pip_count < 0:
        init_no_pip = [3]
    else:
        init_no_pip = [3 + round(pip_count * i) for i in range(10)]
    list_no_pip = np.unique(init_no_pip)

    # w parameter
    init_w = [4,6,8,10,12,14,16,18,20,22,26,30,35,40,50,60,70,80,90,100]
    list_w = [i for i in init_w if i < (0.01*args.max_percent)*len_of_ts]
    if len(list_w) == 0: list_w.append(1)

    best_no_pip = 5
    best_w = 18
    best_ri = -1
    cluster = KMEAN_PISD(parameter=[num_of_classes, best_no_pip, best_w, args.max_iter])

    # Tuning no_pip when fix w = 18
    best_w = min(args.first_w, list_w[-1])
    cluster.set_w(w=best_w)
    print("Tuning no_pip when fix w = %s" % best_w)
    for no_pip in list_no_pip:
        cluster.set_no_pip(no_pip=no_pip)
        cluster.fit(parameter=[train_data, train_labels])
        ri = cluster.leave_one_out_tuning()
        if ri > best_ri:
            best_no_pip = no_pip
            best_ri = ri

        print("fix w: %s - no_pip: %s/%s - c_best_ri: %s - best_noip: %s"
              % (best_w, no_pip, list_no_pip[-1], best_ri, best_no_pip))
        if best_ri == 1:
            break

    # Tuning w when fix no_pip = best_no_pip
    print("Tuning w when fix no_pip = %s" % (best_no_pip))
    cluster.set_no_pip(best_no_pip)
    cluster.fit(parameter=[train_data, train_labels])
    if best_no_pip == 3:
        best_w = 1
    else:
        for w in list_w:
            if best_ri == 1:
                break
            cluster.set_w(w=w)
            ri = cluster.leave_one_out_tuning()
            if ri > best_ri:
                best_w = w
                best_ri = ri
            print("fix no_pip: %s - w: %s/%s - c_ri: %s - best_w: %s" % (best_no_pip, w, list_w[-1], best_ri, best_w))

    print("best no_pip: %s - best w: %s - best_ri: %s" % (best_no_pip, best_w, best_ri))
    cluster.set_no_pip(best_no_pip)
    cluster.set_w(w=best_w)
    cluster.fit(parameter=[store_train_data, store_train_labels])
    test_ri, test_ars, test_nmi = cluster.cluster(parameter=[test_data, test_labels])
    print("dts: %s - no_pip: %s - w: %s - tr_ri: %s - te_ri: %s" % (name, best_no_pip, best_w, best_ri, test_ri))

