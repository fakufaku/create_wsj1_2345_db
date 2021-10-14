# Import packages
import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import scipy as scipy
from scipy.io import wavfile

from config_path import get_paths
from parallel_proc import process
from utils import (ProgressBar, is_clipped, read_source_images,
                   wav_format_to_float)


def check_mix_parallel(
    n_sources, n_microphones, dic, config_path, config, fail_indices
):

    if dic["start"] == 0:
        print(f"Checking mix of {n_sources} sources and {n_microphones} microphones")

    output_path = config_path.output_path

    for subset_key in config_path.subset_list:

        if subset_key != dic["key"]:
            continue

        path = (
            config_path.output_path
            / config_path.db_root
            / config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)
            / f"{subset_key}"
        )

        path_mixinfo_json = os.path.join(path, "mixinfo.json")
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

            wav_snr_mixing = sim_info["wav_snr_mixing"]

            # check that the mix is not clipped
            _, mix = wavfile.read(output_path / sim_info["wav_dpath_mixed_reverberant"])
            if is_clipped(mix):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "clipped",
                        "value": "mix",
                    }
                )

            # check that non of the channels is zero
            if np.any(np.max(np.abs(mix), axis=0) == 0):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "channel zero",
                        "value": "mix",
                    }
                )

            # check anechoic mix is not clipped
            anechoic_images_paths = [
                output_path / p for p in sim_info["wav_dpath_image_anechoic"]
            ]
            anechoic_images = read_source_images(anechoic_images_paths)
            if is_clipped(anechoic_images):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "clipped",
                        "value": "anechoic images",
                    }
                )

            # check that none of the channels is zero
            if np.any(np.max(np.abs(anechoic_images), axis=-1) == 0):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "channel zero",
                        "value": "anechoic images",
                    }
                )

            # check relative power of sources
            images_paths = [
                output_path / p for p in sim_info["wav_dpath_image_reverberant"]
            ]
            reverb_images = read_source_images(images_paths)

            # check that images are not clipped
            if is_clipped(reverb_images):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "clipped",
                        "value": "reverberant images",
                    }
                )

            # check that none of the channels is zero
            if np.any(np.max(np.abs(reverb_images), axis=-1) == 0):
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "channel zero",
                        "value": "reverb images",
                    }
                )

            reverb_images = wav_format_to_float(reverb_images)

            # Check the SNR of the sources with respect to each other
            power_reverberant_images = np.sum(np.square(reverb_images), axis=(1, 2))
            # compute actual SNR of the files
            snr = 10.0 * np.log10(
                power_reverberant_images / power_reverberant_images[0]
            )

            # compute difference with target value
            snr_error = np.max(np.abs(snr - wav_snr_mixing))

            if snr_error >= config["tests"]["snr_tol"]:
                fail_indices.append(
                    {
                        "subset": subset_key,
                        "index": index,
                        "src": n_sources,
                        "mic": n_microphones,
                        "error": "snr",
                        "value": snr_error,
                    }
                )

            if dic["start"] == 0:
                progress_bar.tick()


def check_mix(config, config_path):
    # we use a manager to gather data from different processes
    manager = multiprocessing.Manager()
    fail_indices = manager.list()

    process(
        check_mix_parallel, config, config_path, extra_proc_args=[config, fail_indices]
    )

    # show some of the errors, if any
    if len(fail_indices):
        error_fn = "check_mix_errors.json"

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

    check_mix(config, config_path)
