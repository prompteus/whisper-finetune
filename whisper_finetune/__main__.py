from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import datasets
import typer
from transformers import Seq2SeqTrainingArguments
from transformers.hf_argparser import HfArgumentParser

from whisper_finetune.train import train_model
from whisper_finetune.utils import ModelSize

app = typer.Typer()

DEFAULT_TRAINING_ARGS = dict(
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    evaluation_strategy="steps",
    report_to="wandb",
)


@app.command()
def download_common_voice(
    seed: int = 0,
) -> None:
    ...


@app.command()
def prepare_data(
    seed: int = 0,
) -> None:
    ...


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def train(
    ctx: typer.Context,
    model_name_pretrained: Optional[str] = typer.Option(None),
    dataset_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    dataset_name: str = typer.Option(...),
    noise_songs_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    noise_other_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    cache_dir_models: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, writable=True),
    should_early_stop: bool = typer.Option(True),
    early_stopping_patience: Optional[int] = typer.Option(None),
    wandb_project: Optional[str] = typer.Option(None),
    lang: str = typer.Option(...),
    lang_long: str = typer.Option(...),
    model_size: ModelSize = typer.Option(...),
    transcript_col_name: str = typer.Option("transcription"),
) -> None:

    for subset in ("train", "validation", "test"):
        assert (dataset_dir / subset).exists()
        assert (dataset_dir / subset).is_dir()
        assert any((dataset_dir / subset / f"metadata.{suffix}").exists() for suffix in ("csv", "jsonl"))

    for arg, value in DEFAULT_TRAINING_ARGS.items():
        if arg not in ctx.args:
            ctx.args.extend([f"--{arg}", str(value)])

    dataset = datasets.load_dataset("audiofolder", data_dir=str(dataset_dir))
    assert isinstance(dataset, datasets.DatasetDict)

    training_args_parser = HfArgumentParser(Seq2SeqTrainingArguments)
    training_args = training_args_parser.parse_args_into_dataclasses(ctx.args, return_remaining_strings=True)[0]
    assert isinstance(training_args, Seq2SeqTrainingArguments)

    if wandb_project is None:
        wandb_project = f"whisper-{lang}"

    if model_name_pretrained is None:
        model_name_pretrained = f"openai/whisper-{model_size.value}"
    model_name_finetuned = f"{model_name_pretrained.split('/')[-1]}-{lang}-{dataset_name}"

    if should_early_stop and early_stopping_patience is None:
        raise ValueError("early_stopping_patience must be set if should_early_stop is True")

    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

    train_model(
        model_name_pretrained=model_name_pretrained,
        model_name_finetuned=model_name_finetuned,
        dataset_name=dataset_name,
        dataset=dataset,
        training_args=training_args,
        noise_songs_dir=noise_songs_dir,
        noise_other_dir=noise_other_dir,
        cache_dir_models=cache_dir_models,
        lang=lang,
        lang_long=lang_long,
        model_size=model_size,
        wandb_project_name=wandb_project,
        should_early_stop=should_early_stop,
        early_stopping_patience=early_stopping_patience,
        transcript_col_name=transcript_col_name,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
