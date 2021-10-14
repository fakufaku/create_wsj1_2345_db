import json
import multiprocessing
import os
import sys

import numpy as np


def syswrite(*args):
    for idx in range(len(args)):
        sys.stdout.write(str(args[idx]))
        if idx != len(args) - 1:
            sys.stdout.write(" ")
    sys.stdout.flush()


def get_data_num(n_sources, n_microphones, n_splits, config_path):
    """
    Split a dataset into a number of "splits" to process in parallel
    """
    fname = "wsj1_{}_mix_m{}".format(n_sources, n_microphones)
    out_list = {}
    for subset_key in config_path.subset_list:
        out_list[subset_key] = []
        path_mixinfo_json = os.path.join(
            config_path.output_path,
            config_path.db_root,
            fname,
            subset_key,
            "mixinfo.json",
        )
        with open(path_mixinfo_json, mode="r") as f:
            mixinfo = json.load(f)
        data_qty = len(mixinfo)

        split_size = int(np.ceil(data_qty / n_splits))
        split_start = 0
        while split_start < data_qty:
            split_end = min(split_start + split_size, data_qty)
            out_list[subset_key].append(
                {"key": subset_key, "start": split_start, "end": split_end}
            )
            split_start = split_end

    return out_list


def process(func, config, config_path, n_cpus=None, extra_proc_args=None):

    if not extra_proc_args:
        extra_proc_args = tuple()
    else:
        extra_proc_args = tuple(extra_proc_args)

    for c in config["combinations"]:

        if n_cpus is None:
            n_splits = multiprocessing.cpu_count()
        else:
            n_splits = n_cpus

        dic_list = get_data_num(
            c["sources"], c["mics"], n_splits, config_path=config_path
        )

        for subset_key, sub_dic_list in dic_list.items():
            print(f"Processing {subset_key} in {len(sub_dic_list)} processes")
            processes = []

            for dic in sub_dic_list:
                p = multiprocessing.Process(
                    target=func,
                    args=(c["sources"], c["mics"], dic, config_path) + extra_proc_args,
                )
                # 開始します
                p.start()
                processes.append(p)

            # wait for all processes to finish before continuing
            for p in processes:
                p.join()
