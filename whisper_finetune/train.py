from __future__ import annotations

from pathlib import Path

import datasets
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

import wandb
from whisper_finetune.augment import MyAugment, my_augment_pipeline
from whisper_finetune.metrics import MyStringMetrics
from whisper_finetune.preprocess import DataCollatorSpeechSeq2SeqWithPadding, Preprocessor
from whisper_finetune.utils import SUPPORTED_AUDIO_FORMATS, ModelSize, files_ending_with


def train_model(
    model_name_pretrained: str,
    model_name_finetuned: str,
    dataset_name: str,
    dataset: datasets.DatasetDict,
    training_args: Seq2SeqTrainingArguments,
    noise_songs_dir: Path,
    noise_other_dir: Path,
    cache_dir_models: Path,
    lang: str,
    lang_long: str,
    model_size: ModelSize,
    wandb_project_name: str,
    should_early_stop: bool,
    early_stopping_patience: int | None,
    transcript_col_name: str,
) -> None:

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_name_pretrained,
        cache_dir=cache_dir_models,
    )

    tokenizer = WhisperTokenizer.from_pretrained(
        model_name_pretrained,
        language=lang_long,
        task="transcribe",
        cache_dir=cache_dir_models,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_pretrained,
        cache_dir=cache_dir_models,
    )

    print(f"Scaning {noise_songs_dir} and {noise_other_dir} for noise files.")

    paths_noise_songs = list(files_ending_with(noise_songs_dir, SUPPORTED_AUDIO_FORMATS, deep=True))
    paths_noise_other = list(files_ending_with(noise_other_dir, SUPPORTED_AUDIO_FORMATS, deep=True))

    print(f"Found {len(paths_noise_songs)} songs and {len(paths_noise_other)} other noise files.")

    augment = MyAugment(
        my_augment_pipeline(
            paths_noise_other=paths_noise_other,
            paths_noise_songs=paths_noise_songs,
            seed=42,
        )
    )

    preprocess_train = Preprocessor(tokenizer, feature_extractor, augment_fn=augment)
    preprocess_eval = Preprocessor(tokenizer, feature_extractor)

    collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer, feature_extractor)
    audio_feature = datasets.Audio(sampling_rate=feature_extractor.sampling_rate)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    dataset = dataset.cast_column("audio", audio_feature)
    dataset = dataset.rename_column(transcript_col_name, "transcription")
    dataset["train"].set_transform(preprocess_train)
    dataset["validation"].set_transform(preprocess_eval)
    dataset["test"].set_transform(preprocess_eval)

    wandb_run = wandb.init(
        project=wandb_project_name,
        tags=[lang, model_size.value, dataset_name],
    )
    assert wandb_run is not None

    wandb_run.config.update(
        {
            "lang": lang,
            "model_size": model_size.value,
            "model_name_pretrained": model_name_pretrained,
            "model_name_finetuned": model_name_finetuned,
            "dataset_name": dataset_name,
        }
    )

    callbacks = []
    if should_early_stop:
        assert early_stopping_patience is not None
        early_stopping = EarlyStoppingCallback(early_stopping_patience)
        training_args.load_best_model_at_end = True
        callbacks.append(early_stopping)

    training_args.per_device_train_batch_size = 2
    training_args.remove_unused_columns = False

    metrics = MyStringMetrics(tokenizer)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=metrics.compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
