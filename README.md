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



## Augmentation Preparation
TODO

## Training
TODO

## Evaluation
TODO
