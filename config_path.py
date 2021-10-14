# config_path.py
#
# Describe important paths here
#

import os
from pathlib import Path


class Paths:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_paths(config, original_path, output_path):
    """
    Returns a dict containing all the paths
    """
    original_path = Path(original_path)
    output_path = Path(output_path)

    db_root = Path(config["db_name"])

    max_file_per_folder = 400

    path_raw_wav = db_root / "original/raw_wav"
    path_transcript = db_root / "original/transcript"

    wsj0_root = Path("csr_1/")
    wsj1_root = Path("csr_2_comp/")

    return Paths(
        original_path=original_path,
        output_path=output_path,
        subfolder_fmt="wsj1_{srcs}_mix_m{mics}",
        subset_list=["si284", "dev93", "eval92"],
        subset_n_files={"si284": 37416, "dev93": 503, "eval92": 333},
        max_file_per_folder=max_file_per_folder,
        # DB input
        wsj0_root=wsj0_root,
        wsj1_root=wsj1_root,
        ndx_list={
            "si284": [
                wsj0_root / "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
                wsj1_root / "13-34.1/wsj1/doc/indices/si_tr_s.ndx",
            ],
            "dev93": [wsj1_root / "13-34.1/wsj1/doc/indices/h1_p0.ndx"],
            "eval92": [wsj0_root / "11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx"],
        },
        # DB output
        db_root=db_root,
        raw_wav={"path": path_raw_wav, "metadata_file": "raw_wav.json"},
        transcript={
            "path": path_transcript,
            "metadata_espnet_style": "espnet_style.json",
            "metadata_dot_style": "dot_style.json",
        },
        sph2pipe=output_path / "sph2pipe_v2.5/sph2pipe",
        # noise path
        noise_root="CHiME3/data/audio/16kHz/backgrounds/",
        noise_si284=[
            "BGD_150203_010_CAF",
            "BGD_150203_010_PED",
            "BGD_150203_010_STR",
            "BGD_150203_020_PED",
            "BGD_150204_010_BUS",
            "BGD_150204_020_BUS",
            "BGD_150204_020_CAF",
            "BGD_150211_020_STR",
            "BGD_150211_030_STR",
        ],
        noise_dev93=[
            "BGD_150212_040_STR",
            "BGD_150205_030_PED",
            "BGD_150204_030_CAF",
            "BGD_150204_030_BUS",
        ],
        noise_eval92=[
            "BGD_150205_040_CAF",
            "BGD_150211_040_PED",
            "BGD_150204_040_BUS",
            "BGD_150212_050_STR",
        ],
    )
