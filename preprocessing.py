from __future__ import print_function

import os
import subprocess
from multiprocessing import Pool, cpu_count
from src import reorganize, unwarp_strip_skull

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_path_paras(data_dir, src_dir, dst_dir, data_labels):
    data_src_dir = os.path.join(data_dir, src_dir)
    data_dst_dir = os.path.join(data_dir, dst_dir)
    data_src_paths, data_dst_paths = [], []
    for label in data_labels:
        src_label_dir = os.path.join(data_src_dir, label)
        dst_label_dir = os.path.join(data_dst_dir, label)
        create_dir(dst_label_dir)
        for subject in os.listdir(src_label_dir):
            data_src_paths.append(os.path.join(src_label_dir, subject))
            data_dst_paths.append(os.path.join(dst_label_dir, subject))
    return data_src_paths, data_dst_paths

def process(func, paras):
    pool = Pool(processes=cpu_count())
    pool.map(func, paras)
    return


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data")
    data_labels = ["1", "2"]

    # reorganize()
    # skull_stripping
    data_src_paths, data_dst_paths = get_path_paras(data_dir, "FLAIR", "FLAIRBrain", data_labels)
    paras = zip(data_src_paths, data_dst_paths)
    process(unwarp_strip_skull, paras)