from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import augly.audio.functional as audaugsF
import datasets
import numpy as np
import torch
from _collections_abc import Mapping
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def shrink_splits(
    ds: datasets.DatasetDict,
    split_sizes: dict[str, int | None],
    grow_split: str = "train",
    seed: int = 0,
) -> datasets.DatasetDict:
    """
    Shrink splits to a given size.
    Moves the ommited examples to 'grow_split' split.

    None values in split_sizes are ignored and the split is not modified.
    """

    rng = np.random.default_rng(seed)
    dataset = datasets.DatasetDict()
    moved = []

    for split, size in split_sizes.items():
        if size is not None and len(ds[split]) > size:
            idx_remaining = rng.choice(len(ds[split]), size, replace=False)
            idx_move = np.setdiff1d(np.arange(len(ds[split])), idx_remaining)
            dataset[split] = ds[split].select(idx_remaining)
            moved.append(ds[split].select(idx_move))

    dataset[grow_split] = datasets.concatenate_datasets([ds[grow_split]] + moved)
    return dataset


@dataclass
class Preprocessor:
    tokenizer: WhisperTokenizer
    feature_extractor: WhisperFeatureExtractor
    augment_fn: Callable | None = None
    process: bool = True

    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        for audio in batch["audio"]:
            array = audio["array"]
            rate = audio["sampling_rate"]
            array, rate = audaugsF.to_mono(array, rate)
            audio["array"], audio["sampling_rate"] = array, rate

        if self.augment_fn is not None:
            # modifies in-place
            self.augment_fn(batch)

        batch_size = len(batch["audio"])
        if self.process:
            batch["input_features"] = [None] * batch_size
            for i in range(batch_size):
                array = batch["audio"][i]["array"]
                batch["input_features"][i] = self.feature_extractor(array, sampling_rate=rate).input_features[0]
            batch["labels"] = self.tokenizer(batch["transcription"]).input_ids
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    tokenizer: WhisperTokenizer
    feature_extractor: WhisperFeatureExtractor

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def save_common_voice_to_files(
    dataset: datasets.DatasetDict,
    dataset_dir: Path,
    shrink_test_split: int | None,
    shrink_valid_split: int | None,
    seed: int,
) -> None:
    del dataset["other"]
    del dataset["invalidated"]
    assert set(dataset.keys()) == {"train", "test", "validation"}

    dataset = shrink_splits(
        dataset,
        {
            "test": shrink_test_split,
            "validation": shrink_valid_split,
        },
        grow_split="train",
        seed=seed,
    )

    os.makedirs(dataset_dir, exist_ok=True)

    for split, data in dataset.items():
        os.makedirs(dataset_dir / split)
        for path in tqdm(data["path"], desc=f"Copying {split} audio files"):
            shutil.copy2(path, dataset_dir / split)

    def extract_name(example: Mapping) -> Mapping:
        example["file_name"] = Path(example["file_name"]).name
        return example

    dataset = (
        dataset.remove_columns(["audio"])
        .rename_columns({"sentence": "transcription", "path": "file_name"})
        .map(extract_name)
    )

    for split, ds in dataset.items():
        meta_path = dataset_dir / split / f"metadata.jsonl"
        ds.to_json(meta_path, orient="records", lines=True, force_ascii=False)
