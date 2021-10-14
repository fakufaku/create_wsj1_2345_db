import json
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_source_images(paths):
    # check relative power of sources
    images = []
    for ni, path in enumerate(paths):
        _, image = wavfile.read(path)

        images.append(image)

    images = np.transpose(
        np.stack(images, axis=1), [1, 2, 0],  # (wavelength, Nsource, Nmic)
    )  # -> (Nsource, Nmic, wavelength)

    return images


def write_wav(filename, fs, data):
    parentdir = Path(filename).parent
    os.makedirs(parentdir, exist_ok=True)
    wavfile.write(filename, fs, data)


def is_clipped(signal):
    if signal.dtype == np.int16:
        return signal.max() >= 2 ** 15 - 1 or signal.max() <= -(2 ** 15)
    elif signal.dtype == np.int32:
        return signal.max() >= 2 ** 31 - 1 or signal.max() <= -(2 ** 31)
    elif signal.dtype in [np.float, np.float32, np.float64]:
        return np.max(np.abs(signal)) >= 1.0
    else:
        raise NotImplementedError("Only types int16, int32, or float are supported")


def wav_format_to_int16(signal):
    return (signal * (2 ** 15)).astype(np.int16)


def wav_format_to_float(signal, dtype=np.float64):
    is_16bit = signal.dtype == np.int16
    signal = signal.astype(dtype)
    if is_16bit:
        signal /= 2 ** 15
    return signal


class ExtendedEncoder(json.JSONEncoder):
    """
    A helper class to convert ndarray and Path object
    before saving to JSON file
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.int):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


# Print iterations progress
class ProgressBar:
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    def __init__(
        self,
        total,
        prefix="",
        suffix="",
        decimals=1,
        length=100,
        fill="█",
        print_end="\r",
        refresh_rate=0.001,
    ):

        self.iteration = 0
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        if refresh_rate is not None:
            self.refresh_iteration = max([1, int(refresh_rate * self.total)])
        else:
            self.refresh_iteration = None

    def print(self):
        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (self.iteration / float(self.total))
        )
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + "-" * (self.length - filled_length)
        print(f"\r{self.prefix} |{bar}| {percent}% {self.suffix}", end=self.print_end)
        # Print New Line on Complete
        if self.iteration == self.total:
            print()

    def tick(self, n=1):
        self.iteration += n
        if (
            self.refresh_iteration is None
            or self.iteration % self.refresh_iteration == 0
        ):
            self.print()

    def update(self, iteration):
        self.iterations = iteration
        self.print()
