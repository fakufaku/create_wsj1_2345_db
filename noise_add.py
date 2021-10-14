# Import packages
import argparse
import json
import multiprocessing
import os
import wave
from pathlib import Path

import numpy as np
import scipy as scipy
from scipy.io import wavfile

# Import original sources
from config_path import get_paths
from parallel_proc import process
from utils import (ExtendedEncoder, ProgressBar, is_clipped,
                   wav_format_to_float, wav_format_to_int16, write_wav)


class NoiseFile:
    org_microphones = [1, 2, 3, 4, 5, 6]

    def __init__(self, config_path, noise_folder, org_mics=None):
        if org_mics is None:
            self.org_mics = NoiseFile.org_microphones
        else:
            self.org_mics = org_mics

        self.fids = []

        self.duration = None

        for m in self.org_mics:
            file_name = os.path.join(
                config_path.original_path / config_path.noise_root,
                "{}.CH{}.wav".format(noise_folder, m),
            )
            fid = wave.open(file_name)

            if not self.duration:
                self.duration = fid.getnframes()
            else:
                assert fid.getnframes() == self.duration

            assert fid.getnchannels() == 1
            assert fid.getframerate() == 16000

            self.fids.append(fid)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for fid in self.fids:
            fid.close()

    def __len__(self):
        return self.duration

    def get_samples(self, pos, n_samples):
        arr = []
        for fid in self.fids:
            fid.setpos(pos)

            if fid.getsampwidth() == 2:
                dtype = np.int16
            elif fid.getsampwidth() == 4:
                dtype = np.int32
            else:
                raise ValueError("only supports 16bit and 32 bit files")
            arr.append(np.frombuffer(fid.readframes(n_samples), dtype=dtype))
        return np.column_stack(arr)


# 雑音データをメモリに展開
def load_noise_data(config_path):
    noise_dict = {}
    noise_dict["si284"] = {"noise_path": config_path.noise_si284}
    noise_dict["dev93"] = {"noise_path": config_path.noise_dev93}
    noise_dict["eval92"] = {"noise_path": config_path.noise_eval92}

    return noise_dict


def select_noise(subset_key, n_microphones, duration, noise_dict, config_path):
    org_microphones = NoiseFile.org_microphones.copy()
    np.random.shuffle(org_microphones)

    noise_path = np.random.choice(noise_dict[subset_key]["noise_path"])

    # Open the noise file and read only the necessary part
    with NoiseFile(config_path, noise_path, org_mics=org_microphones) as noise_file:
        n_samples = len(noise_file)
        # randomize the start location of the sample
        st = np.random.randint(0, n_samples - duration)

        # read the samples
        noise_data = noise_file.get_samples(st, duration)

        # remove extra channels and convert to float
        mix_noise = wav_format_to_float(noise_data[:, :n_microphones], dtype=np.float64)

    return mix_noise


def noise_add_one(
    config_path, sim_info, subset_key, n_sources, noise_dict, wav_upper_limit
):

    # fs
    fs = sim_info["wav_frame_rate_mixed"]
    # sample, channel
    _, reverberant_mix_data = wavfile.read(
        config_path.output_path / sim_info["wav_dpath_mixed_reverberant"]
    )
    reverberant_mix_data = wav_format_to_float(reverberant_mix_data, dtype=np.float64)

    # 雑音データをトル
    n_samples, n_microphones = reverberant_mix_data.shape
    mix_noise = select_noise(
        subset_key, n_microphones, n_samples, noise_dict, config_path
    )

    # scaling factor to adjust the SNR
    scaling_factor = np.sqrt(
        10 ** (sim_info["noise_snr"] / 10)
        * np.sum(np.square(mix_noise))
        / np.sum(np.square(reverberant_mix_data))
    )
    reverberant_noise_mix = mix_noise / scaling_factor + reverberant_mix_data

    # if the signal clips after mixing the noise, we might need to
    # readjust the scale of all the files produced so far
    if is_clipped(reverberant_noise_mix):

        # collect all the necessary paths
        op = config_path.output_path
        paths = [op / sim_info["wav_dpath_mixed_reverberant"]]
        for s in range(n_sources):
            paths.append(op / sim_info["wav_dpath_image_reverberant"][s])
            paths.append(op / sim_info["wav_dpath_image_anechoic"][s])

        # read in the files
        signals = dict(
            zip(paths, [wav_format_to_float(wavfile.read(p)[1]) for p in paths])
        )

        # the maximum of the reverberant signal, which is clipping
        noise_mix_max = np.max(np.abs(reverberant_noise_mix))

        # this should allow to make all the signals non-clipping without
        # affecting the SNR values defined
        scaling_factor = wav_upper_limit / noise_mix_max

        # adjust the scale of the noisy mix
        reverberant_noise_mix *= scaling_factor

        # adjust the scale and save all the other signals
        for path, data in signals.items():
            write_wav(path, fs, wav_format_to_int16(data * scaling_factor))

    write_wav(
        config_path.output_path / sim_info["wav_mixed_noise_reverb"],
        fs,
        wav_format_to_int16(reverberant_noise_mix),
    )


def choose_and_add_noise(
    n_sources, n_microphones, dic, config_path, config, write_sync_lock
):

    # folder_name
    fname = config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)

    # load the noise data information
    noise_dict = load_noise_data(config_path)

    if dic["start"] == 0:
        print(f"Add noise for {n_sources} sources and {n_microphones} microphones")

    for subset_key in config_path.subset_list:

        # Check if directory for output exists
        path_mixinfo_json = os.path.join(
            config_path.db_root, fname, subset_key, "mixinfo.json"
        )
        with open(config_path.output_path / path_mixinfo_json, mode="r") as f:
            mixinfo = json.load(f)

        str_len = max([len(x) for x in config_path.subset_list])
        prefix = "{:" + str(str_len) + "}"
        progress_bar = ProgressBar(
            dic["end"] - dic["start"], prefix=prefix.format(subset_key)
        )

        mixinfo_noise = {}

        for pair_index, (pair_id, sim_info) in enumerate(mixinfo.items()):

            if (
                dic["key"] == subset_key
                and pair_index >= dic["start"]
                and pair_index < dic["end"]
            ):

                # here we use the seed that we picked for this instance
                rng_state = np.random.get_state()
                np.random.seed(sim_info["seed"])

                # choose SNR at random
                sim_info["noise_snr"] = np.random.uniform(
                    *config["mixinfo_parameters"]["noise"]["snr_range"]
                )

                # path to save noisy reverberant data
                subfolder = f"{int(pair_id) // config_path.max_file_per_folder:03d}"
                path_wav_noise_reverberant = os.path.join(
                    config_path.db_root,
                    fname,
                    subset_key,
                    "wav_mixed_noise_reverb",
                    subfolder,
                    pair_id + ".wav",
                )

                mixinfo[pair_id]["wav_mixed_noise_reverb"] = path_wav_noise_reverberant

                noise_add_one(
                    config_path,
                    sim_info,
                    subset_key,
                    n_sources,
                    noise_dict,
                    config["mixinfo_parameters"]["wav_upper_limit"],
                )

                # add the information to the new mixinfo dict
                mixinfo_noise[pair_id] = sim_info

                # restore RNG state
                np.random.set_state(rng_state)

                # we only show the progress bar from one process
                # i.e., the one that starts at zero
                if dic["start"] == 0:
                    progress_bar.tick()

        if dic["key"] == subset_key:
            path_mixinfo_noise_json = (
                config_path.output_path
                / config_path.db_root
                / fname
                / subset_key
                / "mixinfo_noise.json"
            )

            with write_sync_lock:

                # if was created by a different process, we will append to it
                if path_mixinfo_noise_json.exists():
                    with open(path_mixinfo_noise_json, "r") as f:
                        existing_mixinfo = json.load(f)
                    existing_mixinfo.update(mixinfo_noise)
                    mixinfo_noise = existing_mixinfo

                # save the json metadata file
                with open(path_mixinfo_noise_json, mode="w") as f:
                    json.dump(mixinfo_noise, f, indent=4, cls=ExtendedEncoder)


def noise_add(config, config_path):
    # we use a lock to synchronize writing to the output files
    write_sync_lock = multiprocessing.Lock()

    # run in several processing
    process(
        choose_and_add_noise,
        config,
        config_path,
        extra_proc_args=[config, write_sync_lock],
    )


# main
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

    noise_add(config, config_path)
