# Import packages
import argparse
import json
import multiprocessing
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pa
import scipy as scipy
import scipy.io.wavfile

# Import original sources
from config_path import get_paths
from parallel_proc import process
from utils import (ProgressBar, wav_format_to_float, wav_format_to_int16,
                   write_wav)


# PyroomacousticsでRIRのSimulationを実施
def acoustic_simulation(config_path, sim_info, config, check_empty_channels=False):
    # fs
    fs = sim_info["wav_frame_rate_mixed"]
    # 残響時間
    rt60 = sim_info["rir_info_t60"]
    # 部屋の大きさ
    room_dim = np.array(sim_info["rir_info_room_dimension"])
    # 残響時間から逆算
    e_absorption, max_order = pa.inverse_sabine(rt60, room_dim)

    R = np.array(sim_info["rir_info_microphone_position"])
    R = R.T
    # 音源位置の情報
    speaker_locations = np.array(sim_info["rir_info_speaker_position"])
    speaker_locations = speaker_locations.T
    n_sources = speaker_locations.shape[1]

    # 各音源をシミュレーションに追加する
    clean_data_path = sim_info["wav_dpath_original"]
    reverberant_conv_data = []
    anechoic_conv_data = []
    raw_data = []

    temp_dtype = np.float64

    for s in range(n_sources):

        # 最初にReverberantの場合について
        room = pa.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=e_absorption)
        room_anechoic = pa.ShoeBox(room_dim, fs=fs, max_order=0)
        # 用いるマイクロホンアレイの情報を設定する
        room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
        room_anechoic.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
        _, clean_data = scipy.io.wavfile.read(
            config_path.output_path / clean_data_path[s]
        )

        clean_data = wav_format_to_float(clean_data, dtype=temp_dtype)

        zero_mean = config["mixinfo_parameters"].get("remove_mean_sources", False)
        if zero_mean:
            clean_data = clean_data - clean_data.mean()

        filled_clean_data = np.zeros(
            (sim_info["wav_n_samples_mixed"]), dtype=temp_dtype
        )
        clean_begin = sim_info["wav_offset"][s]
        clean_end = sim_info["wav_offset"][s] + sim_info["wav_n_samples_original"][s]
        filled_clean_data[clean_begin:clean_end] = clean_data
        norm = np.sqrt(np.average(np.square(filled_clean_data)))
        filled_clean_data /= norm
        raw_data.append(filled_clean_data.copy())

        add_success = False
        while not add_success:
            try:
                room.add_source(speaker_locations[:, s], signal=filled_clean_data)
                room_anechoic.add_source(
                    speaker_locations[:, s], signal=filled_clean_data
                )
                add_success = True
            except Exception:
                add_success = False

        # room.add_source(speaker_locations[:, s], signal=filled_clean_data)
        room.simulate(snr=None)
        # room_anechoic.add_source(speaker_locations[:, s], signal=filled_clean_data)
        room_anechoic.simulate(snr=None)
        n_sample = np.shape(filled_clean_data)[0]
        # 畳み込んだ波形を取得する(チャンネル、サンプル）

        reverberant_signals = room.mic_array.signals[:, :n_sample].copy()
        anechoic_signals = room_anechoic.mic_array.signals[:, :n_sample].copy()
        weight = np.sqrt(
            10 ** (np.float(sim_info["wav_snr_mixing"][s]) / 10.0)
        ) / np.sqrt(np.sum(np.square(reverberant_signals)))
        reverberant_conv_data.append(weight * reverberant_signals)
        anechoic_conv_data.append(weight * anechoic_signals)

    reverberant_mix = np.sum(reverberant_conv_data, axis=0)

    # Rescale mixed sources so as the maximum value is MAX * 0.9
    upper_limit = config["mixinfo_parameters"]["wav_upper_limit"]
    mixed_max = np.max(
        [
            np.max(np.abs(data))
            for data in [reverberant_mix, reverberant_conv_data, anechoic_conv_data]
        ]
    )
    scaling_factor_mixed = upper_limit / mixed_max

    reverberant_mix *= scaling_factor_mixed

    # save all the files
    for s in range(n_sources):
        # Write source images files
        write_wav(
            config_path.output_path / sim_info["wav_dpath_image_reverberant"][s],
            fs,
            wav_format_to_int16(reverberant_conv_data[s].T * scaling_factor_mixed),
        )
        write_wav(
            config_path.output_path / sim_info["wav_dpath_image_anechoic"][s],
            fs,
            wav_format_to_int16(anechoic_conv_data[s].T * scaling_factor_mixed),
        )

    # Write mixed .wav
    write_wav(
        config_path.output_path / sim_info["wav_dpath_mixed_reverberant"],
        fs,
        wav_format_to_int16(reverberant_mix.T),
    )


def simulate_mix(n_sources, n_microphones, dic, config_path, config):
    # folder_name
    fname = config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)
    output_path = config_path.output_path

    if dic["start"] == 0:
        print(
            f"Simulate and mix for {n_sources} sources and {n_microphones} microphones"
        )

    subset_key = dic["key"]

    # Check if directory for output exists
    wav_reverberant_unmixed_output_dir = os.path.join(
        output_path, config_path.db_root, fname, subset_key, "wav_image_reverb"
    )
    if not os.path.isdir(wav_reverberant_unmixed_output_dir):
        os.makedirs(wav_reverberant_unmixed_output_dir, exist_ok=True)

    wav_anechoic_unmixed_output_dir = os.path.join(
        output_path, config_path.db_root, fname, subset_key, "wav_image_anechoic"
    )
    if not os.path.isdir(wav_anechoic_unmixed_output_dir):
        os.makedirs(wav_anechoic_unmixed_output_dir, exist_ok=True)

    wav_reverberant_output_dir = os.path.join(
        output_path, config_path.db_root, fname, subset_key, "wav_mixed_reverb"
    )
    if not os.path.isdir(wav_reverberant_output_dir):
        os.makedirs(wav_reverberant_output_dir, exist_ok=True)

    wav_source_output_dir = os.path.join(
        output_path, config_path.db_root, fname, subset_key, "wav_source"
    )
    if not os.path.isdir(wav_source_output_dir):
        os.makedirs(wav_source_output_dir, exist_ok=True)

    path_mixinfo_json = os.path.join(
        output_path, config_path.db_root, fname, subset_key, "mixinfo.json"
    )
    with open(path_mixinfo_json, mode="r") as f:
        mixinfo = json.load(f)

    str_len = max([len(x) for x in config_path.subset_list])
    prefix = "{:" + str(str_len) + "}"
    progress_bar = ProgressBar(
        dic["end"] - dic["start"], prefix=prefix.format(subset_key)
    )

    pair_index = 0
    for pair_id in mixinfo:
        # Simulation with pyroomacoustics

        if pair_index >= dic["start"] and pair_index < dic["end"]:
            acoustic_simulation(config_path, mixinfo[pair_id], config)

            # we only show the progress bar from one process
            # i.e., the one that starts at zero
            if dic["start"] == 0:
                progress_bar.tick()

        pair_index += 1


def mix(config, config_path):
    # this is only done on one pair to get the size of the dataset
    process(simulate_mix, config, config_path, extra_proc_args=[config])


"""
def process_errors(config, config_path, error_file):

    with open(error_file, "r") as f:
        errors = json.load(f)

    for error in errors:
        n_sources = error["src"]
        n_microphones = error["mic"]
        subset_key = error["subset"]
        pair_id = error["index"]

        fname = config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)
        output_path = config_path.output_path

        path_mixinfo_json = os.path.join(
            output_path, config_path.db_root, fname, subset_key, "mixinfo.json"
        )
        with open(path_mixinfo_json, mode="r") as f:
            mixinfo = json.load(f)
"""


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

    mix(config, config_path)
