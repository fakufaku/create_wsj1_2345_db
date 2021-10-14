A dataset for Multichannel Blind Source Separation and Dereverberation
======================================================================

## Generate the dataset

### Preliminaries

#### Environment

Using [anaconda](https://www.anaconda.com/products/individual) should make it
easy to install all the dependencies and reproduce the dataset.
After installing anaconda do the following.

```bash
git clone git@github.com:fakufaku/create_wsj1_2345_db.git
cd create_wsj1_2345_mix_spatialized
conda env create -f environment.yml
conda activate wsj1_2345_db
```

#### Original datasets

The script assumes that you have the following datasets available and
stored in a directory that we'll assume is named `<original_datasets_dir>`

* [WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A) stored in a folder named `csr_1`
* [WSJ1](https://catalog.ldc.upenn.edu/LDC94S13A) stored in a folder named `csr_2_comp`
* [CHIME3](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2015/) (noise only) stored in a folder named `CHIME3`

For WSJ0 and WSJ1, we assume their respective folders contain subfolders named
after each of the DVDs.  The detailed original datasets directory structure is
shown in detail [here](#original-datasets-struct).


### Create the whole dataset from scratch

```bash
python ./make_dataset.py config.json <original_datasets_dir> <output_dir>
```

### Create the dataset step-by-step

```bash
# convert WSJ1 nist format to regular wav
python ./make_raw_wav.py config.json <original_datasets_dir> <output_dir>

# get the text transcription from the audio
python ./get_trans.py config.json <original_datasets_dir> <output_dir>

# create the mix metadata
python ./create_mixinfo.py config.json <original_datasets_dir> <output_dir>

# simulate propagation and mix the audio, then check
python ./mix.py config.json <original_datasets_dir> <output_dir>
python ./check_mix.py config.json <original_datasets_dir> <output_dir>

# add noise to all the mixtures, then check
python ./noise_add.py config.json <original_datasets_dir> <output_dir>
python ./check_noisy_mix.py config.json <original_datasets_dir> <output_dir>
```

### Configure the dataset

The dataset generation is controlled by a JSON file like the following

```json
{
    "db_name": "wsj1_2345_db",
    "combinations":
    [
        { "mics": 2, "sources": 2, "seed": 639872833 },
        { "mics": 3, "sources": 3, "seed": 312393873 },
        { "mics": 4, "sources": 4, "seed": 739853286 }
    ],
    "mixinfo_parameters": {
        "room": { "l": [5, 10], "w": [5, 10], "h": [3, 4], "t60": [0.2, 0.6] },
        "array": {
            "xy_jittering": 0.2,
            "z": [1, 2],
            "radius": [0.075, 0.125],
            "min_dist_mics": 0.05
        },
        "speaker": {
            "xy_square": 3.0,
            "z": [1.5, 2.0],
            "min_dist_array": 0.5,
            "min_dist_speaker": 1.0,
            "snr": [-5, 5]
        },
        "noise": {
            "snr_range": [10, 30]
        },
        "wav_upper_limit": 0.9,
        "remove_mean_sources": true
    },
    "tests": {
        "snr_tol": 0.5
    }
}
```


## Changelog

* Fix all seeds, one seed per sample (because of multiprocessing)
* Use only numpy.random
* SNR is computed with respect to reverberant signals
* Corrects placement of microphones on/in a sphere
* Adds noise SNR in the mixinfo file
* Moved the definition of all the simulation parameters to the configurationfile
* The output wav file format has been changed from float32 to int16


## Difference with MERL dataset

Some of the differences with [wsj0-2mix/3mix](https://www.merl.com/demos/deep-clustering) dataset.

1. Handles more sources
2. RIR Generator is changed to [pyroomacoustics](https://github.com/LCAV/pyroomacoustics).
3. Adds noise from CHiME3 background dataset
4. Only up to 6 channels because this is the number of channels in CHiME3


## Appendix: Orignal datasets directory structure
<a name="original-dataset-struct"></a>

```
<original_datasets_dir>
+-- csr_1
|   +-- 11-1.1
|   +-- 11-10.1
|   +-- 11-11.1
|   +-- 11-12.1
|   +-- 11-13.1
|   +-- 11-14.1
|   +-- 11-15.1
|   +-- 11-2.1
|   +-- 11-3.1
|   +-- 11-4.1
|   +-- 11-5.1
|   +-- 11-6.1
|   +-- 11-7.1
|   +-- 11-8.1
|   +-- 11-9.1
|   +-- file.tbl
|   +-- readme.txt
+-- csr_2_comp
|   +-- 13-1.1
|   +-- 13-10.1
|   +-- 13-11.1
|   +-- 13-12.1
|   +-- 13-13.1
|   +-- 13-14.1
|   +-- 13-15.1
|   +-- 13-16.1
|   +-- 13-17.1
|   +-- 13-18.1
|   +-- 13-19.1
|   +-- 13-2.1
|   +-- 13-20.1
|   +-- 13-21.1
|   +-- 13-22.1
|   +-- 13-23.1
|   +-- 13-24.1
|   +-- 13-25.1
|   +-- 13-26.1
|   +-- 13-27.1
|   +-- 13-28.1
|   +-- 13-29.1
|   +-- 13-3.1
|   +-- 13-30.1
|   +-- 13-31.1
|   +-- 13-32.1
|   +-- 13-33.1
|   +-- 13-34.1
|   +-- 13-4.1
|   +-- 13-5.1
|   +-- 13-6.1
|   +-- 13-7.1
|   +-- 13-8.1
|   +-- 13-9.1
+-- CHIME3
|   +-- data
|       +-- audio
|           +-- 16kHz
|               +-- backgrounds
|                   +-- BGD_150203_010_CAF.CH1.wav
|                   +-- BGD_150203_010_CAF.CH2.wav
|                   +-- BGD_150203_010_CAF.CH3.wav
|                   +-- ...
```

License
-------

2020-2021 (c) Robin Scheibler, Masahito Togami, Masaya Wake, LINE Corporation

Code released under [MIT License](https://opensource.org/licenses/MIT).
