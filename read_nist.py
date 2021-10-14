import numpy as np
from scipy.io import wavfile

TEST_FILE_NIST = (
    "/mnt/shared_storage/datasets/csr_1/11-10.1/wsj0/sd_tr_s/00f/00fo040b.wv2"
)
TEST_FILE_WAV = "./bss_speech_dataset/data/channels2_room16_mix.wav"

_nist_type_decoder = {
    "-s": str,
    "-i": int,
    "-f": float,
}


def _read_nist_decode_header_field(field):
    key, t, value = field.split(maxsplit=2)

    if len(t) == 2:
        n_chars = len(value)
    else:
        n_chars = int(t[2:])
    value = _nist_type_decoder[t[:2]](value[:n_chars])

    return key, value


def _read_nist_dtype(n_bytes, byte_format_str):

    if n_bytes == 2:
        dt = np.dtype(np.int16)
        if byte_format_str == "01":
            return dt
        elif byte_format_str == "10":
            return dt.newbyteorder(">")
        else:
            raise NotImplementedError(
                f"n_bytes={n_bytes} byte_format_str={byte_format_str}"
            )
    else:
        raise NotImplementedError(
            f"n_bytes={n_bytes} byte_format_str={byte_format_str}"
        )


def _read_nist_header(f):

    end_marker = "end_head"
    newline = b"\n"

    raw_header = b""

    # get read of first line
    c = ""
    while c != newline:
        c = f.read(1)
        raw_header += c

    # read number of bytes in header on second line
    n_bytes_header_str = b""
    c = ""
    while c != newline:
        c = f.read(1)
        n_bytes_header_str += c

    # concatenate to header
    raw_header += n_bytes_header_str

    # convert the byte string to an integer
    n_bytes_header = int(n_bytes_header_str)

    # read-in the rest of the header
    while len(raw_header) < n_bytes_header:
        raw_header += f.read(1)

    # convert to ASCII
    raw_header = raw_header.decode("ascii")

    lines = raw_header.split("\n")

    header = {}

    for line in lines[2:]:

        if line.startswith(end_marker):
            break
        else:
            key, value = _read_nist_decode_header_field(line)
            header[key] = value

    fmt_marker = lines[0][:-1]

    return fmt_marker, header


def read_audio_nist(filename):

    with open(filename, "rb") as f:
        fmt_marker, header = _read_nist_header(f)
        data = f.read()

    # sampling frequency
    fs = header["sample_rate"]

    # number of channels
    if "numChannels" in header and "channel_count" not in header:
        n_channels = header["numChannels"]
    elif "channel_count" in header:
        n_channels = header["channel_count"]

    if "channels_interleaved" in header:
        channels_interleaved = header["channels_interleaved"] == "TRUE"
    else:
        channels_interleaved = True

    # byte format
    n_bytes = header["sample_n_bytes"]
    byte_fmt_str = header["sample_byte_format"]
    dtype = _read_nist_dtype(n_bytes, byte_fmt_str)

    # total number of bytes
    n_samples = min(header["sample_count"], len(data)) // n_bytes

    n_samples = (n_samples // n_channels) * n_channels

    shape = (n_samples // n_channels, n_channels)
    if not channels_interleaved:
        order = "F"
    else:
        order = "C"

    audio = np.frombuffer(data, dtype=dtype, count=n_samples).reshape(
        shape, order=order
    )

    if n_channels == 1:
        audio = audio[:, 0]

    return fs, audio


reader = {
    "NIST": read_audio_nist,
    "RIFF": wavfile.read,
}


def read_audio(filename):

    with open(filename, "rb") as f:
        fmt = f.read(4).decode("ascii")

    return reader[fmt](filename)


if __name__ == "__main__":

    fs, audio = read_audio(TEST_FILE_NIST)
    # read_audio(TEST_FILE_WAV)
