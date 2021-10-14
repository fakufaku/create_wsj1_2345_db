# get_trans.py
#
# This script
#  -- get transcripts (.dot style) and output into a single file
#  -- convert .dot style transcripts into ESPnet style transcripts and output into a single file
#

# Import packages
import argparse
import glob
import json
import os
import re
from pathlib import Path

# Import original sources
from config_path import get_paths
from utils import ExtendedEncoder


# Original function
def get_extended_utterance_id_from_path(path):
    pattern = ".*(wsj[01]).*(.._(tr|dt|et)_..?).*(.{3})(.{5})\..*"
    _m = re.match(pattern, path)
    return _m.group(1) + "_" + _m.group(2) + "_" + _m.group(4) + _m.group(5)


def fix_transcript(tt):
    # Extract our designated texts from dot-stlyed transcripts
    # 1. capitalize
    tt = tt.upper()
    # 2. adopt '
    tt = re.sub(r"\\'", "'", tt)
    # 3. adopt .point and %percent
    tt = re.sub(r"\\.POINT", "POINT", tt)
    tt = re.sub(
        r"\\.PERCENT", "PERCENT", tt
    )  # note that some tags are misspelled as \.PERCENT
    # 4. delete other \ tags
    tt = re.sub(r"\\\S*", "", tt)
    # 5. delete ~, ., : and *
    tt = re.sub(r"[~.:*]", "", tt)
    # 6. delete [/] and [<>] tags
    tt = re.sub(r"\[([/<][A-Za-z0-9]+|[A-Za-z0-9]+[/>])\]", "", tt)
    # 7. adopt <> words
    tt = re.sub(r"[<>]", "", tt)
    # 8. substitute other [] tags into <NOISE>
    tt = re.sub(r"\[.*\]", "<NOISE>", tt)
    # 9. delete dupricated spaces
    tt = re.sub(r"\s{2,}", " ", tt)
    # 10. delete spaces before/after the sentence
    tt = tt.strip()
    return tt


def get_trans(config, config_path):

    if (config_path.output_path / config_path.transcript["path"]).exists():
        print("Original raw directory already exists. Skip.")
        return

    # Get filelist of .dot and dict to search, using extended utterance id
    wsj0_root = config_path.original_path / config_path.wsj0_root
    wsj1_root = config_path.original_path / config_path.wsj1_root
    dot_list = glob.glob(str(wsj0_root) + "/**/*.dot", recursive=True) + glob.glob(
        str(wsj1_root) + "/**/*.dot", recursive=True
    )
    dot_dict = {}
    for dot_path in dot_list:
        dot_dict[get_extended_utterance_id_from_path(dot_path)] = dot_path

    # For each subset,
    for subset_key in config_path.subset_list:
        print(subset_key)

        # Check if directory for output exists
        transcript_output_dir = (
            config_path.output_path / config_path.transcript["path"] / subset_key
        )
        if not os.path.isdir(transcript_output_dir):
            os.makedirs(transcript_output_dir)

        # Reading raw_wav.json
        path_raw_wav_json = (
            config_path.raw_wav["path"]
            / subset_key
            / config_path.raw_wav["metadata_file"]
        )
        with open(config_path.output_path / path_raw_wav_json, mode="r") as f:
            raw_wav_dict = json.load(f)

        transcript_dict_dot = {}
        transcript_dict_espnet = {}

        # For each raw wav,
        for utterance_id in raw_wav_dict:
            extended_utterance_id = get_extended_utterance_id_from_path(
                raw_wav_dict[utterance_id]["original_wv1_path"]
            )
            # Read corresponding .dot file
            dot_path = dot_dict[extended_utterance_id[:-2] + "00"]
            with open(dot_path, mode="r") as f:
                dot_data = f.read()

            # Search corresponding transcript and reformat it into ESPnet style
            pattern = r"^(.+)\s+\(" + utterance_id + r"\)$"
            m = re.search(
                pattern, dot_data, flags=re.MULTILINE
            )  # Since dot_data contains \n
            dot_style_transcript = m.group(1)
            espnet_style_transcript = fix_transcript(dot_style_transcript)

            # Add result into dict
            transcript_dict_dot[utterance_id] = dot_style_transcript
            transcript_dict_espnet[utterance_id] = espnet_style_transcript

        # Output as dot-style file
        path_transcript_dot = (
            transcript_output_dir / config_path.transcript["metadata_dot_style"]
        )
        with open(path_transcript_dot, mode="w") as f:
            json.dump(transcript_dict_dot, f, indent=4, cls=ExtendedEncoder)

        # Output as espnet-style file
        path_transcript_espnet = (
            transcript_output_dir / config_path.transcript["metadata_espnet_style"]
        )
        with open(path_transcript_espnet, mode="w") as f:
            json.dump(transcript_dict_espnet, f, indent=4, cls=ExtendedEncoder)


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

    # open the config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # get all the paths
    config_path = get_paths(config, args.original_dataset_paths, args.output_path)

    get_trans(config, config_path)
