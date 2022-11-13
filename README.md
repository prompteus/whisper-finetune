# Whisper finetune

This project is made for easy finetuning (training) of Whisper speech-to-text models.

## Installation
TODO

## Dataset Preparation
The dataset is expected have the following structure:
```
<dataset_root>
|-- test
|   |-- <filename_01>.wav
|   |-- <filename_02>.wav
|   |-- ...
|   `-- metadata.jsonl
|-- train
|   |-- <filename_01>.wav
|   |-- <filename_02>.wav
|   |-- ...
|   `-- metadata.jsonl
`-- validation
|   |-- <filename_01>.wav
|   |-- <filename_02>.wav
|   |-- ...
    `-- metadata.jsonl
```
Other audio formats, like `mp3` or `ogg` are also supported.
Other metadata formats, like `csv` are also supported.
(For both, see huggingface datasets `load_dataset` function).

Metadata must contain at least two fields:
- `file_name` - the name of the related audio file, including file extension
- `transcription` - the ground truth transcription

The name of the transcription column can differ, but in that case, must be passed to the training script as:

```whisper-finetune train ... --transcript-col-name=<your_column_name>```

:warning: &nbsp; **All files are expected to be shorter than 30s. If they are not,
they get truncated, but the transcription is not (because where should it be cut?)
As a result, the model would be trained to hallucinate words not spoken in the audio.**


## Downloading Mozilla Common Voice dataset
In case you don't want to use your own dataset, you can use Mozilla Common Voice.
The package comes with a cli command to download it into an expected format.

```
whisper-finetune download-common-voice \
    --dataset-dir ./data/common_voice \
    --cache-dir ./data \
    --lang cs \
    --shrink-test-split 4000 \
    --shrink-valid-split 2000 \
```


## Augmentation Preparation
`whisper-finetune` comes with a prepared audio augmentation pipeline.

It consists of several audio effects provided by [https://github.com/facebookresearch/AugLy](augly) library.
It also adds songs/environmental noise to the audio. Unfortunately, the audio files need to be partly downloaded manually.
Here are the scripts I use for preparing augmentation data

### Songs
[FMA small](https://github.com/mdeff/fma), 8000 tracks of 30s, 8 balanced genres

```
mkdir -p ./data

wget -O ./data/FMA-small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip 
unzip ./data/FMA-small.zip -d ./data/
```

### Environment noise
[ESC-50](https://github.com/karolpiczak/ESC-50), dataset for classification of environment noise.
(50 types divided into categories: animals, natural soundscapes, human non-speech, domestic sounds,	urban noises)
The dataset is not large, but balanced.

```
wget -O ./data/esc_50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
unzip ./data/esc_50.zip -d ./data/
mv ./data/ESC-50-master/audio ./data/ESC-50
rm -r ./data/ESC-50-master
rm ./data/esc_50.zip
```

[ESC-US](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT), larger and more diverse dataset of noises, but it is not guaranteed to be balanced (or representative). Unfortunatelly, it needs to be downloaded manualy (because of license agreement). 2 parts of it should be enough. More than that would defeat the purpose of downloading the smaller balanced ESC-50 anyways.

Download the parts manually from the website, place them in the data folder and run:
```
mkdir -p ./data/ESC
find ./data/ -name ESC-US-*.tar.gz -print0 | parallel -0 tar -xvzf {} -C ./data/ESC/ESC-US
find ./data/ -name ESC-US-*.tar.gz -print0 | parallel -0 rm {}
```

### Check augmentations file tree

Now, your data folder should look like this:

```
data
|-- ESC
|   |-- ESC-50
|   `-- ESC-US
|-- FMA-small
...
```

Some of these contain subdirectories, but it does not matter, the training script will find the audio files recursively.



## Training
TODO

## Evaluation
TODO
