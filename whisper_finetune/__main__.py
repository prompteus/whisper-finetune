from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Optional

import datasets
import typer
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from transformers.hf_argparser import HfArgumentParser

import wandb
from whisper_finetune.preprocess import save_common_voice_to_files
from whisper_finetune.train import train_model
from whisper_finetune.utils import ModelSize

app = typer.Typer()

DEFAULT_TRAINING_ARGS = dict(
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    report_to="wandb",
)


@app.command()
def download_common_voice(
    dataset_dir: Path = typer.Option(..., file_okay=False, dir_okay=True, readable=True),
    cache_dir: Path = typer.Option(..., file_okay=False, dir_okay=True, readable=True),
    overwrite_if_exists: bool = typer.Option(False),
    lang: str = typer.Option(...),
    hf_dataset_name: str = typer.Option("mozilla-foundation/common_voice_11_0"),
    shrink_test_split: Optional[int] = typer.Option(None),
    shrink_valid_split: Optional[int] = typer.Option(None),
    seed: int = 0,
) -> None:
    dataset_dir = dataset_dir / lang
    if dataset_dir.exists():
        if overwrite_if_exists:
            typer.echo(f"Overwriting existing dataset at {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            typer.echo(f"Dataset already exists at {dataset_dir}")
            return

    dataset = datasets.load_dataset(hf_dataset_name, lang, use_auth_token=True, cache_dir=cache_dir)

    save_common_voice_to_files(
        dataset,
        dataset_dir,
        shrink_test_split=shrink_test_split,
        shrink_valid_split=shrink_valid_split,
        seed=seed,
    )

    print(f"All files and metadata copied to {dataset_dir}.")


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
    output_root_dir: Optional[Path] = typer.Option(None, file_okay=False, dir_okay=True, writable=True),
    should_early_stop: bool = typer.Option(False),
    early_stopping_patience: Optional[int] = typer.Option(None),
    wandb_project: Optional[str] = typer.Option(None),
    lang: str = typer.Option(...),
    lang_long: str = typer.Option(...),
    model_size: ModelSize = typer.Option(...),
    transcript_col_name: str = typer.Option("transcription"),
    seed: int = typer.Option(0),
) -> None:

    for subset in ("train", "validation", "test"):
        assert (dataset_dir / subset).exists()
        assert (dataset_dir / subset).is_dir()
        assert any((dataset_dir / subset / f"metadata.{suffix}").exists() for suffix in ("csv", "jsonl"))

    for arg, value in DEFAULT_TRAINING_ARGS.items():
        if arg not in ctx.args:
            ctx.args.extend([f"--{arg}", str(value)])

    if model_name_pretrained is None:
        model_name_pretrained = f"openai/whisper-{model_size.value}"
    model_name_finetuned = f"{model_name_pretrained.split('/')[-1]}-{lang}--{dataset_name}"

    if "--output_dir" in ctx.args and output_root_dir is not None:
        raise ValueError(
            "Cannot specify both --output_dir and --output-root-dir. --output-root-dir uses automatic naming."
        )

    if "--output_dir" not in ctx.args and output_root_dir is None:
        raise ValueError(
            "Must specify either --output_dir or --output-root-dir. --output-root-dir uses automatic naming."
        )

    if wandb_project is None:
        wandb_project = f"whisper-{lang}"
    wandb_run = wandb.init(project=wandb_project)

    if output_root_dir is not None:
        output_dir = str(output_root_dir / model_name_finetuned / wandb_run.name)
        ctx.args.extend([f"--output_dir", output_dir])

    dataset = datasets.load_dataset("audiofolder", data_dir=str(dataset_dir))
    assert isinstance(dataset, datasets.DatasetDict)

    training_args_parser = HfArgumentParser(Seq2SeqTrainingArguments)
    training_args, remaining_args = training_args_parser.parse_args_into_dataclasses(
        ctx.args, return_remaining_strings=True
    )

    assert isinstance(training_args, Seq2SeqTrainingArguments)
    assert isinstance(training_args, TrainingArguments)

    print("TRAINING ARGS: ")
    print(training_args)
    print("\n\n")

    print("REMAINING ARGS:")
    print(remaining_args)
    print("\n\n")

    if should_early_stop and early_stopping_patience is None:
        raise ValueError("early_stopping_patience must be set if should_early_stop is True")

    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

    train_model(
        model_name_pretrained=model_name_pretrained,
        model_name_finetuned=model_name_finetuned,
        dataset_name=dataset_name,
        dataset=dataset,
        wandb_run=wandb_run,
        training_args=training_args,
        noise_songs_dir=noise_songs_dir,
        noise_other_dir=noise_other_dir,
        cache_dir_models=cache_dir_models,
        lang=lang,
        lang_long=lang_long,
        model_size=model_size,
        should_early_stop=should_early_stop,
        early_stopping_patience=early_stopping_patience,
        transcript_col_name=transcript_col_name,
        seed=seed,
    )

    print("Model trained.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
