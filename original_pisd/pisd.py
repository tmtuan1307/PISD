import sys
import argparse
import numpy as np
import copy
import ucr_data_loader as data_loader
import config as config
from nn_pisd_model import NN_PISD
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='', type=str, help="You can use dataset_name directly, please "
                                                                             "see in the config.py for all dataset name")
    parser.add_argument("--dataset_pos", default=2, type=int, help="Or You can just use dataset pos in the"
                                                                            "config.py file")
    parser.add_argument("--first_w", default=18, type=int, help="The first window size for tuning num_of_pip parameter")
    args = parser.parse_args()

    all_dataset_name = config.ALL_DATASET_NAME
    classifier_flag = "PISD"

    first_w = args.first_w
    if args.dataset_name != "":
        name = args.dataset_name
    else:
        name = all_dataset_name[args.dataset_pos]

    print("Starting: algorithm: %s - dataset: %s" % (classifier_flag, name))
    train_data, train_labels, test_data, test_labels = data_loader.load_dataset(name, config.DATASET_PATH)
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
    list_w = [i for i in init_w if i < 0.2*len_of_ts]
    if len(list_w) == 0: list_w.append(1)

    best_no_pip = -1
    best_w = -1
    best_train_error_rate = np.inf

    classifier = NN_PISD([best_no_pip, best_w, len_of_ts])
    best_w = min(first_w, max(list_w))
    classifier.set_w(w=best_w)
    print("Tuning no_pip when fix w = %s" % best_w)
    for no_pip in list_no_pip:
        classifier.set_no_pip(no_pip=no_pip)
        classifier.fit(parameter=[train_data, train_labels])
        err, error_list = classifier.leave_one_out_tuning(parameter=[best_train_error_rate])
        if err < best_train_error_rate:
            best_train_error_rate, best_no_pip = err, no_pip
        train_data = data_loader.sort_data_by_error_list(train_data, error_list)
        train_labels = data_loader.sort_data_by_error_list(train_labels, error_list)
        train_data, train_labels = np.asarray(train_data), np.asarray(train_labels)
        print("fix w: %s - no_pip: %s/%s - c_best_train_err: %s - best_noip: %s"
              % (best_w, no_pip, list_no_pip[-1], best_train_error_rate, best_no_pip))
        if best_train_error_rate == 0:
            break

    # Tuning w when fix no_pip = best_no_pip
    print("Tuning w when fix no_pip = %s" % (best_no_pip))
    classifier.set_no_pip(best_no_pip)
    classifier.fit(parameter=[train_data, train_labels])
    if best_no_pip == 3:
        best_w = 1
    else:
        for w in list_w:
            if best_train_error_rate == 0:
                break
            classifier.set_w(w=w)
            err, error_list = classifier.leave_one_out_tuning(parameter=[best_train_error_rate])
            if err < best_train_error_rate:
                best_train_error_rate, best_w = err, w
            classifier.sort_pips_piss_due_to_error_list(error_list)
            print("fix no_pip: %s - w: %s/%s - c_best_train_err: %s - best_w: %s"
                  % (best_no_pip, w, list_w[-1], best_train_error_rate, best_w))

    print("best no_pip: %s - best w: %s - best_train_err: %s"
          % (best_no_pip, best_w, best_train_error_rate))
    classifier.set_no_pip(best_no_pip)
    classifier.set_w(w=best_w)
    classifier.fit(parameter=[store_train_data, store_train_labels])
    test_error_rate = round(classifier.predict(parameter=[test_data, test_labels]), 3)
    print("dts: %s - no_pip: %s - w: %s - train_err: %s - test_err: %s"
          % (name, best_no_pip, best_w, best_train_error_rate,test_error_rate))


