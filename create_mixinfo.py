# create_mixinfo.py
#  -- This script works for up to 6 sources
# This script
#  -- makes pairs of two utterances
#      -- avoiding (A.) same speaker's utterances are mixed
#                  (B.) same pair (e.g. A-B / B-A) is made
#  -- decides offset of shorter utterance
#  -- decides SNR
#  -- decides parameters such as 3-D position of speakers and microphones
#  -- makes RIR using py-RIR-Generator
#  -- outputs these information into JSON file
#      -- conbining transcripts (espnet-style and espnet-style token)
#
# Import packages
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics

# Import original sources
from config_path import get_paths
from parallel_proc import syswrite
from utils import ExtendedEncoder, ProgressBar


def choose_groups_at_random_no_repetition(
    config_path, raw_wav_dict, subset_key, group_size
):

    if group_size == 1:
        return [[uid] for uid in raw_wav_dict.keys()]

    n_groups = len(raw_wav_dict)
    utterance_ids = list(raw_wav_dict.keys())

    selected_groups = []
    sorted_groups = set()

    str_len = max([len(x) for x in config_path.subset_list])
    prefix = "{:" + str(str_len) + "}"
    progress_bar = ProgressBar(
        n_groups, prefix=prefix.format(subset_key), suffix="choose sources"
    )

    for utt_id in utterance_ids:

        while True:
            group = [utt_id] + np.random.choice(
                utterance_ids, size=group_size - 1
            ).tolist()

            # test that no speaker is repeated
            utts = set()
            spkrs = set()
            for el in group:
                utts.add(el)
                spkrs.add(raw_wav_dict[el]["speaker_id"])

            # if there is a repetition of either a speaker or utterance,
            # then the set size will be smaller than the desired group size.
            # in that case, we just try again
            if len(spkrs) < group_size or len(utts) < group_size:
                continue

            # now we test that this group has not been generated before
            sorted_group = tuple(sorted(group))

            if sorted_group in sorted_groups:
                continue
            else:
                sorted_groups.add(sorted_group)
                selected_groups.append(group)
                break

        progress_bar.tick()

    return selected_groups


# centerから半径r内にある点をランダムに選ぶ
def draw_a_point_inside_a_ball(center, r):
    # unit vector
    u = np.random.randn(center.shape[0])
    u *= np.random.uniform(0, r) / np.linalg.norm(u)
    return center + u


# マイクロホン間隔が.0.05mよりも近いとダメ
def check_microphone_pos(microphone_pos, min_dist=0.05):
    for a, mic_a in enumerate(microphone_pos):
        for b, mic_b in enumerate(microphone_pos[a + 1 :]):
            if np.linalg.norm(mic_a - mic_b) < min_dist:
                return False
    return True


def check_speaker_pos(
    array_center, speaker_pos, min_dist_speaker=1.0, min_dist_array=0.5
):
    for s1, s1_pos in enumerate(speaker_pos):
        # Speaker-ArrayCenter
        if np.linalg.norm(array_center - s1_pos) < min_dist_array:
            return False
        for s2_pos in speaker_pos[s1 + 1 :]:
            # Speaker-Speaker
            if np.linalg.norm(s1_pos - s2_pos) < min_dist_speaker:
                return False
    return True


# 音源数毎のフォルダをつくる
def create_mixinfo_one(n_sources, n_microphones, config_path, config):

    # folder_name
    fname = config_path.subfolder_fmt.format(srcs=n_sources, mics=n_microphones)

    for subset_key in config_path.subset_list:

        output_path = config_path.output_path

        # あればつくらない
        os.makedirs(output_path / config_path.db_root, exist_ok=True)

        # Check if directory for output exists
        dir_rir_reverb = config_path.db_root / fname / subset_key / "rir/reverberant"

        dir_rir_anechoic = config_path.db_root / fname / subset_key / "rir/anechoic"

        dir_wav_mixed_reverb = (
            config_path.db_root / fname / subset_key / "wav_mixed_reverb"
        )
        dir_wav_image_reverb = (
            config_path.db_root / fname / subset_key / "wav_image_reverb"
        )
        dir_wav_image_anechoic = (
            config_path.db_root / fname / subset_key / "wav_image_anechoic"
        )
        dir_wav_source = config_path.db_root / fname / subset_key / "wav_source"

        mixinfo = {}

        # Reading raw_wav.json
        path_raw_wav_json = (
            config_path.raw_wav["path"]
            / subset_key
            / config_path.raw_wav["metadata_file"]
        )
        with open(output_path / path_raw_wav_json, mode="r") as f:
            raw_wav_dict = json.load(f)

        # Reading transcript
        path_tran_espnet = (
            config_path.transcript["path"]
            / subset_key
            / config_path.transcript["metadata_espnet_style"]
        )
        with open(output_path / path_tran_espnet, mode="r") as f:
            tran_espnet = json.load(f)

        # Choose pairs
        #  -- 1. Ininitialize pair table
        """
        pair_table = {utterance_id_key: "None" for utterance_id_key in raw_wav_dict}

        candidate_lists = []
        for source_index in range(1, n_sources):
            candidate_list = [utterance_id_key for utterance_id_key in raw_wav_dict]
            candidate_lists.append(candidate_list)

        str_len = max([len(x) for x in config_path.subset_list])
        prefix = "{:" + str(str_len) + "}"
        progress_bar = ProgressBar(
            len(pair_table), prefix=prefix.format(subset_key), suffix="choose sources"
        )

        #  -- 2. For each utterance, set proper partner
        index = 0
        for mix1_utterance_id in pair_table:
            trial_counter = 0
            index = index + 1
            while True:
                source_ids = [mix1_utterance_id]
                utterance_id_candidates = [
                    np.random.choice(candidate_list)
                    for candidate_list in candidate_lists
                ]
                source_ids.extend(utterance_id_candidates)

                trial_counter += 1

                if trial_counter > 1000:
                    print("Warning: trial_counter exploding")

                is_reject = False
                # idの重複をチェック
                for id1 in range(n_sources):
                    for id2 in range(id1 + 1, n_sources):
                        if (
                            raw_wav_dict[source_ids[id1]]["speaker_id"]
                            == raw_wav_dict[source_ids[id2]]["speaker_id"]
                        ):
                            # reject
                            for candidate_list, utterance_id_candidate in zip(
                                candidate_lists, utterance_id_candidates
                            ):
                                # candidate_list.append(utterance_id_candidate)
                                is_reject = True

                # 組み合わせが既にあるかをチェック
                for id in utterance_id_candidates:
                    temp_source_ids = source_ids.copy()
                    temp_source_ids.remove(id)

                    compare_ids = pair_table[id]
                    if compare_ids != "None":
                        temp_source_ids.sort()
                        temp_compare_ids = compare_ids.copy()
                        temp_compare_ids.sort()
                        is_exist_combination = True
                        for id1, id2 in zip(temp_source_ids, temp_compare_ids):
                            if id1 != id2:
                                is_exist_combination = False

                        if is_exist_combination:
                            # reject
                            for candidate_list, utterance_id_candidate in zip(
                                candidate_lists, utterance_id_candidates
                            ):
                                # candidate_list.append(utterance_id_candidate)
                                is_reject = True

                # approval
                if not is_reject:
                    # pair approval
                    pair_table[mix1_utterance_id] = utterance_id_candidates
                    # Making next pair
                    break

            progress_bar.tick()
        """
        selected_groups = choose_groups_at_random_no_repetition(
            config_path, raw_wav_dict, subset_key, n_sources
        )
        # convert to pair table format
        pair_table = {}
        for group in selected_groups:
            pair_table[group[0]] = group[1:]

        # number of digits necessary to write all the indices in strings
        # of fixed length. minimum length is 5
        str_len = max([len(x) for x in config_path.subset_list])
        prefix = "{:" + str(str_len) + "}"
        num_digits = max(5, int(np.ceil(np.log10(len(pair_table)))))
        format_string = "{:0>" + str(num_digits) + "}"

        progress_bar = ProgressBar(
            len(pair_table), prefix=prefix.format(subset_key), suffix="generate rooms"
        )

        for pair_index, pair_key in enumerate(pair_table):
            # Print progress
            pair_id = format_string.format(pair_index)
            subfolder = f"{pair_index // config_path.max_file_per_folder:03d}"

            subdir_rir_reverb = dir_rir_reverb / subfolder
            subdir_rir_anechoic = dir_rir_anechoic / subfolder
            subdir_wav_mixed_reverb = dir_wav_mixed_reverb / subfolder
            subdir_wav_image_reverb = dir_wav_image_reverb / subfolder
            subdir_wav_image_anechoic = dir_wav_image_anechoic / subfolder
            subdir_wav_source = dir_wav_source / subfolder

            # Re-get pair information
            utterance_ids = [pair_key]
            utterance_ids.extend(pair_table[pair_key])

            # Decide offset for the shorter utterance
            pair_n_samples_array = [
                raw_wav_dict[utterance_id]["n_samples"]
                for utterance_id in utterance_ids
            ]

            max_samples = np.max(pair_n_samples_array)

            offset_samples = [
                np.random.randint(0, max_samples - n_samples + 1)
                for n_samples in pair_n_samples_array
            ]

            # Decide parameters for RIR
            # This ranges are equal to wsj0_mix (MERL)
            #  ** VALUES are NOT EQUAL!!!
            # 1. Room dimension
            p = config["mixinfo_parameters"]

            # room dimension
            room_l = np.random.uniform(*p["room"]["l"])
            room_w = np.random.uniform(*p["room"]["w"])
            room_h = np.random.uniform(*p["room"]["h"])
            room_dimension = np.array([room_l, room_w, room_h])

            # 2. Microphone array's center position
            array_xy_jittering = p["array"]["xy_jittering"]
            array_center_x = room_l / 2 + np.random.uniform(
                -array_xy_jittering, array_xy_jittering
            )
            array_center_y = room_w / 2 + np.random.uniform(
                -array_xy_jittering, array_xy_jittering
            )
            array_center_z = np.random.uniform(*p["array"]["z"])
            array_center = np.array([array_center_x, array_center_y, array_center_z])

            # 3. Each microphone's position
            microphone_pos = np.zeros((n_microphones, 3))
            array_r = np.random.uniform(*p["array"]["radius"])

            #  -- a. First 2 microphones are on a sphere and symmetric
            unit_vec = np.random.randn(3)
            unit_vec /= np.linalg.norm(unit_vec)
            microphone_pos[0] = array_center + unit_vec * array_r
            if n_microphones > 1:
                microphone_pos[1] = array_center - unit_vec * array_r

            #  -- b. Next 2 microphones are inside a sphere and dist. are at least 0.05m
            if n_microphones > 2:
                while True:
                    microphone_pos[2] = draw_a_point_inside_a_ball(
                        array_center, array_r
                    )
                    if n_microphones > 3:
                        microphone_pos[3] = draw_a_point_inside_a_ball(
                            array_center, array_r
                        )
                    if check_microphone_pos(
                        microphone_pos[: min(n_microphones, 4)],
                        min_dist=p["array"]["min_dist_mics"],
                    ):
                        break

            #  -- c. Final 4 microphones are inside a sphere without any other conditions.
            for n in range(4, n_microphones):
                microphone_pos[n] = draw_a_point_inside_a_ball(array_center, array_r)

            # 4. Each speaker's position
            speaker_pos = [None for source in range(n_sources)]
            speaker_pos = np.zeros((n_sources, 3))
            #  -- Dist. between two speakers is at least 1.0m
            #  -- Dist. between a speaker and microphone array's center is at least 0.5m
            while True:
                xy_offset = p["speaker"]["xy_square"] / 2.0
                # horizontal location is within a square around the array
                for n in range(2):
                    speaker_pos[:, n] = np.random.uniform(
                        array_center[n] - xy_offset,
                        array_center[n] + xy_offset,
                        size=n_sources,
                    )
                speaker_pos[:, 2] = np.random.uniform(
                    *p["speaker"]["z"], size=n_sources
                )
                if check_speaker_pos(
                    array_center,
                    speaker_pos,
                    min_dist_speaker=p["speaker"]["min_dist_speaker"],
                    min_dist_array=p["speaker"]["min_dist_array"],
                ):
                    break

            # 5. T60 (unit: s)
            t60 = np.random.uniform(*p["room"]["t60"])

            # 6. Path
            path_rir_reverberant = os.path.join(subdir_rir_reverb, pair_id + ".rir")
            path_rir_anechoic = os.path.join(subdir_rir_anechoic, pair_id + ".rir")

            # 7. Other
            fs = raw_wav_dict[utterance_ids[0]]["frame_rate"]  # 16000

            # Decide SNR
            pair_snr = [0]
            pair_snr.extend(
                [np.random.uniform(*p["speaker"]["snr"]) for n in range(n_sources - 1)]
            )

            # Set output information
            path_wav_reverberant = subdir_wav_mixed_reverb / f"{pair_id}.wav"
            path_wav_reverberant_unmixed = [
                subdir_wav_image_reverb / f"{pair_id}_{s}.wav" for s in range(n_sources)
            ]
            path_wav_anechoic_unmixed = [
                subdir_wav_image_anechoic / f"{pair_id}_{s}.wav"
                for s in range(n_sources)
            ]
            path_wav_source = [
                subdir_wav_source / f"{pair_id}_{s}.wav" for s in range(n_sources)
            ]
            mixed_wav_n_samples = int(np.max(pair_n_samples_array))

            # Mini wsj1_2mix
            if (
                ("--miniset" in sys.argv)
                and (pair_index % 10 != 0)
                and (subset_key == "si284")
            ):
                # Skip
                pair_index += 1
                continue

            # we also pick one seed to generate random data for this sample
            # (seeds have to be 32 bit numbers)
            seed = np.random.randint(2 ** 32)

            # Store mixinfo
            mixinfo[pair_id] = {
                "data_id": pair_id,
                "seed": seed,
                # path for created wav data
                "wav_dpath_mixed_reverberant": path_wav_reverberant,
                "wav_dpath_image_reverberant": path_wav_reverberant_unmixed,
                "wav_dpath_image_anechoic": path_wav_anechoic_unmixed,
                "wav_dpath_source": path_wav_source,
                # transcript
                "transcript_espnet": [
                    tran_espnet[utterance_id] for utterance_id in utterance_ids
                ],
                # figures about mixing
                "wav_n_samples_mixed": mixed_wav_n_samples,
                "wav_frame_rate_mixed": fs,
                "wav_snr_mixing": pair_snr,
                "wav_offset": offset_samples,
                # information about source 1
                "utterance_id": utterance_ids,
                "wav_dpath_original": [
                    raw_wav_dict[utterance_id]["raw_wav_path"]
                    for utterance_id in utterance_ids
                ],
                "speaker_id": [
                    raw_wav_dict[utterance_id]["speaker_id"]
                    for utterance_id in utterance_ids
                ],
                "wav_n_samples_original": pair_n_samples_array,
                # figures about RIR
                "rir_info_room_dimension": room_dimension,
                "rir_info_array_center": array_center,
                "rir_info_array_radius": array_r,
                "rir_info_microphone_position": microphone_pos,
                "rir_info_speaker_position": speaker_pos,
                "rir_info_t60": t60,
                # path for RIR data
                "rir_dpath_reverberant": path_rir_reverberant,
                "rir_dpath_anechoic": path_rir_anechoic,
            }

            progress_bar.tick()

            pair_index += 1

        # Output mixinfo
        dir_mixinfo = config_path.db_root / fname / subset_key
        os.makedirs(output_path / dir_mixinfo, exist_ok=True)
        path_mixinfo = output_path / dir_mixinfo / "mixinfo.json"
        with open(path_mixinfo, mode="w") as f:
            json.dump(mixinfo, f, indent=4, cls=ExtendedEncoder)

        syswrite("completed\n")

        if "--skiprir" in sys.argv:
            print("Making RIR is skipped.")
            continue


def check_mixinfo(n_sources, n_microphones, config_path):
    # folder_name
    fname = "wsj1_{}_mix_m{}".format(n_sources, n_microphones)
    for subset_key in config_path.subset_list:
        path_mixinfo_json = os.path.join(
            config_path.output_path,
            config_path.db_root,
            fname,
            subset_key,
            "mixinfo.json",
        )
        with open(path_mixinfo_json, mode="r") as f:
            mixinfo = json.load(f)

        str_len = max([len(x) for x in config_path.subset_list])
        prefix = "{:" + str(str_len) + "}"
        progress_bar = ProgressBar(len(mixinfo), prefix=prefix.format(subset_key))

        id_list = []
        for info in mixinfo:
            # 部屋の大きさ
            room_dim = np.array(mixinfo[info]["rir_info_room_dimension"])
            speaker_locations = np.array(mixinfo[info]["rir_info_speaker_position"])
            speaker_locations = speaker_locations.T
            n_sources = np.shape(speaker_locations)[1]
            for source in range(n_sources):
                if (
                    speaker_locations[0, source] >= room_dim[0]
                    or speaker_locations[0, source] <= 0
                    or speaker_locations[1, source] >= room_dim[1]
                    or speaker_locations[1, source] <= 0
                    or speaker_locations[2, source] >= room_dim[2]
                    or speaker_locations[2, source] <= 0
                ):
                    print("room", room_dim)
                    print("speaker", speaker_locations[:, source])

            utterance_ids = mixinfo[info]["utterance_id"]
            temp_ids = utterance_ids.copy()
            temp_ids.sort()
            for id1 in range(n_sources):
                for id2 in range(id1 + 1, n_sources):
                    if id1 == id2:
                        print("error")
            id_str = "".join(temp_ids)

            if id_str in id_list:
                print("error")
            id_list.append(id_str)

            progress_bar.tick()


def create_mixinfo(config, config_path):
    for c in config["combinations"]:
        # set the RNG seed for this specific combination
        rng_state = np.random.get_state()
        np.random.seed(c["seed"])

        print(f"Creating mixinfo for {c['sources']} sources and {c['mics']} mics")
        create_mixinfo_one(
            n_sources=c["sources"],
            n_microphones=c["mics"],
            config_path=config_path,
            config=config,
        )

        print("Checking the generated mixinfo")
        check_mixinfo(c["sources"], n_microphones=c["mics"], config_path=config_path)

        # restore the state of the RNG
        np.random.set_state(rng_state)


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

    create_mixinfo(config, config_path)
