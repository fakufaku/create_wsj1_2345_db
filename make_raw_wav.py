# make_wav.py
#
# This script
#  -- converts .wv1 (NIST) into .wav (RIFF) using sph2pipe
#  -- make filelists for .wav
#

# Import packages
import argparse
import json
import os
import re
from pathlib import Path

from scipy.io import wavfile

from audio import read as read_audio

# Import original sources
from config_path import get_paths
from utils import ExtendedEncoder, write_wav


# Original functions
def line2path(config_path, line_text, subset_name):
    # Notice: We need to
    #          -- substitude discname XX_X_X into XX-X.X
    #          -- remove data inside 11-2.1/wsj0/si_tr_s/401
    #          -- be careful about path in file start with /wsjX OR wsjX
    pattern = r"(\d+)_(\d+)_(\d+): */?([\w/]*)(wsj[01])([\w/]*)([0-9a-z]{3})([0-9a-z]{5})(\.wv1|)"
    m = re.match(pattern, line_text.lower())
    if m is None:
        # Path is not written in this line
        return False
    else:
        wsj_pattern = m.group(5)
        if wsj_pattern == "wsj0":
            full_input_path = os.path.join(
                config_path.wsj0_root,
                m.group(1) + "-" + m.group(2) + "." + m.group(3),
                m.group(4) + m.group(5) + m.group(6) + m.group(7) + m.group(8) + ".wv1",
            )
        elif wsj_pattern == "wsj1":
            full_input_path = os.path.join(
                config_path.wsj1_root,
                m.group(1) + "-" + m.group(2) + "." + m.group(3),
                m.group(4) + m.group(5) + m.group(6) + m.group(7) + m.group(8) + ".wv1",
            )
        else:
            # Something is wrong for this line
            return False

        if "11-2.1/wsj0/si_tr_s/401" in full_input_path:
            # We need to remove this from si284 or si84
            return False

        full_output_path = (
            config_path.raw_wav["path"]
            / subset_name
            / (m.group(7) + m.group(8) + ".wav")
        )
        return [full_input_path, full_output_path, m.group(7), m.group(7) + m.group(8)]


def make_raw_wav(config, config_path):

    if (config_path.output_path / config_path.raw_wav["path"]).exists():
        print("Original raw directory already exists. Skip.")
        return

    for subset_key in config_path.subset_list:
        print(f"{subset_key}: started")
        # Check if directory for output exists
        raw_wav_output_dir = (
            config_path.output_path / config_path.raw_wav["path"] / subset_key
        )
        if not os.path.isdir(raw_wav_output_dir):
            os.makedirs(raw_wav_output_dir, exist_ok=True)

        raw_wav_dict = {}
        global_file_index = 0
        for path_ndx in config_path.ndx_list[subset_key]:
            # Read .ndx file
            with open(config_path.original_path / path_ndx, mode="r") as f:
                f_lines = f.readlines()

            # Get paths for input.wv1 and output.wav
            for line in f_lines:
                wav_info = line2path(config_path, line.rstrip("\r\n"), subset_key)
                if not wav_info:
                    continue
                input_path, output_path, speaker_id, utterance_id = wav_info

                # create a subfolder to avoid having too many files in a single folder
                subfolder_idx = global_file_index // config_path.max_file_per_folder
                parent_dir = output_path.parent
                filename = output_path.name
                output_path = parent_dir / f"{subfolder_idx:03d}/{filename}"
                global_file_index += 1

                # read input, write output
                frame_rate, audio = read_audio(config_path.original_path / input_path)
                write_wav(config_path.output_path / output_path, frame_rate, audio)
                wav_samples = audio.shape[0]

                # Add info into dict
                raw_wav_dict[utterance_id] = {
                    "raw_wav_path": output_path,
                    "original_wv1_path": input_path,
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "n_samples": wav_samples,
                    "frame_rate": frame_rate,
                }

        # Dump info dict into json
        path_raw_wav_json = (
            config_path.raw_wav["path"]
            / subset_key
            / config_path.raw_wav["metadata_file"]
        )
        with open(config_path.output_path / path_raw_wav_json, mode="w") as f:
            # the extended encoder allows to save numpy arrays, values and Path objects to JSON
            json.dump(raw_wav_dict, f, indent=4, cls=ExtendedEncoder)

        # Subset finished
        print(
            f"{subset_key}: finished converting "
            f"{len(raw_wav_dict)}/{config_path.subset_n_files[subset_key]} files"
        )
        if len(raw_wav_dict) != config_path.subset_n_files[subset_key]:
            print(
                "Error: the number of files actually "
                "converted is different from expected!"
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

    config_path = get_paths(
        config=config,
        original_path=args.original_dataset_paths,
        output_path=args.output_path,
    )

    make_raw_wav(config, config_path)
