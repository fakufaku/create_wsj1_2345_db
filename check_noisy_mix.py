# Import packages
import argparse
import json
import multiprocessing
import os
from pathlib import Path

import _pickle as cPickle
import numpy as np
import scipy as scipy
import scipy.io.wavfile

from config_path import get_paths
from parallel_proc import process
from utils import ProgressBar, is_clipped, wav_format_to_float


def check_noisy_mix_parallel(
    n_sources, n_microphones, dic, config_path, config, fail_indices
):

    if dic["start"] == 0:
        print(
            f"Checking noisy mix of {n_sources} sources and {n_microphones} microphones"
        )

    for subset_key in config_path.subset_list:

        if subset_key != dic["key"]:
            continue

        path = (
            config_path.output_path
            / config_path.db_root
            / config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)
            / f"{subset_key}"
        )

        path_mixinfo_json = os.path.join(path, "mixinfo_noise.json")
        with open(path_mixinfo_json, mode="r") as f:
            mixinfo = json.load(f)

        str_len = max([len(x) for x in config_path.subset_list])
        prefix = "{:" + str(str_len) + "}"
        progress_bar = ProgressBar(
            dic["end"] - dic["start"], prefix=prefix.format(subset_key)
        )

        for n, (index, sim_info) in enumerate(mixinfo.items()):

            if n < dic["start"] or dic["end"] <= n:
                continue

            # the target snr
            snr_target = sim_info["noise_snr"]

            # read the noisy mix
            _, reverberant_noisy_mix_data = scipy.io.wavfile.read(
                config_path.output_path / sim_info["wav_mixed_noise_reverb"]
            )

            # check for clipping (important to do this *before* converting to float
            if is_clipped(reverberant_noisy_mix_data):
                fail_indices.append(
                    {
                        "n_src": n_sources,
                        "n_mic": n_microphones,
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "clipped",
                        "value": "noisy reverberant mix",
                    }
                )

            # convert to float
            reverberant_noisy_mix_data = wav_format_to_float(reverberant_noisy_mix_data)

            # read the clean mix
            _, reverberant_mix_data = scipy.io.wavfile.read(
                config_path.output_path / sim_info["wav_dpath_mixed_reverberant"]
            )
            reverberant_mix_data = wav_format_to_float(reverberant_mix_data)

            # compute actual SNR of the files
            snr_est = 10.0 * np.log10(
                np.sum(np.square(reverberant_mix_data))
                / np.sum(np.square(reverberant_noisy_mix_data - reverberant_mix_data))
            )

            # compute difference with target value
            snr_error = np.max(np.abs(snr_est - snr_target))

            if snr_error >= config["tests"]["snr_tol"]:
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "snr": {
                            "expected": snr_target,
                            "obtained": snr_est.tolist(),
                            "error": snr_error,
                        },
                        "error": "snr",
                        "value": snr_error,
                    }
                )

            if dic["start"] == 0:
                progress_bar.tick()


def check_noisy_mix(config, config_path):
    # we use a manager to gather data from different processes
    manager = multiprocessing.Manager()
    fail_indices = manager.list()

    process(
        check_noisy_mix_parallel,
        config,
        config_path,
        extra_proc_args=[config, fail_indices],
    )

    # show some of the errors, if any
    if len(fail_indices):
        error_fn = "check_noisy_mix_errors.json"

        print(f"There were {len(fail_indices)} errors. For example:",)
        for i, error in enumerate(fail_indices):
            print(f"  - {error}")
            if i > 9:
                break
        print(f"The full log of errors is saved in {error_fn}")

        # also save to a file for further processing
        with open(error_fn, "w") as f:
            json.dump(list(fail_indices), f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates all the configuration files")
    parser.add_argument("config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "original_dataset_paths",
        type=Path,
        help="Path to folders containing original datasets",
    )
    parser.add_argument(
        "output_path", type=Path, help="Path to destination folder for the output"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # get all the paths
    config_path = get_paths(config, args.original_dataset_paths, args.output_path)

    check_noisy_mix(config, config_path)
