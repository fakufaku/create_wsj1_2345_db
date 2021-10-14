import argparse
import datetime
import json
import time
from pathlib import Path

from check_mix import check_mix
from check_noisy_mix import check_noisy_mix
from config_path import get_paths
from create_mixinfo import create_mixinfo
from get_trans import get_trans
from make_raw_wav import make_raw_wav
from mix import mix
from noise_add import noise_add

steps = {
    "raw": {"description": "Convert raw to wav", "func": make_raw_wav, "depends": []},
    "trans": {
        "description": "Get the transcripts",
        "func": get_trans,
        "depends": ["raw"],
    },
    "mixinfo": {
        "description": "Create the mix metadata files",
        "func": create_mixinfo,
        "depends": ["trans"],
    },
    "mix": {
        "description": "Simulate reverb and record mixes",
        "func": mix,
        "depends": ["mixinfo"],
    },
    "check_mix": {
        "description": "Basic checks on the simulated mixtures",
        "func": check_mix,
        "depends": ["mix"],
    },
    "noise_add": {
        "description": "Add noise to the mixture data",
        "func": noise_add,
        "depends": ["mix"],
    },
    "check_noisy_mix": {
        "description": "Basic checks on the noisy mixtures",
        "func": check_noisy_mix,
        "depends": ["check_noisy_mix"],
    },
}

steps_order = [
    "raw",
    "trans",
    "mixinfo",
    "mix",
    "check_mix",
    "noise_add",
    "check_noisy_mix",
]


def sec_to_str(seconds):
    return str(datetime.timedelta(seconds=seconds))


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

    runtimes = {}

    print(f"Starting generation of the dataset {config['db_name']}")

    for n, skey in enumerate(steps_order):
        # start counting time
        t_start = time.perf_counter()

        print(f"Step {n}: {steps[skey]['description']}")
        steps[skey]["func"](config, config_path)

        # stop counting time
        t_end = time.perf_counter()
        runtimes[skey] = t_end - t_start
        time_str = sec_to_str(runtimes[skey])
        print(f"Step {n} finished in {time_str}")
        print()

    # print some runtime statistics at the end
    print("Summary:")
    total_time = 0
    for n, (skey, rt) in enumerate(runtimes.items()):
        print(f"  - Step {n} {skey} finished in {sec_to_str(rt)}")
        total_time += rt
    print(f"Total time spent: {sec_to_str(total_time)}")
