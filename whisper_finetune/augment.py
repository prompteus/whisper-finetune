from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import augly.audio as audaugs
import augly.audio.functional as audaugsF
import librosa
import numpy as np


class RandomTransform(audaugs.transforms.BaseTransform):
    def __init__(
        self,
        seed: int = 0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p)
        self.random_gen = np.random.default_rng(seed)


class ApplyRandArgs(RandomTransform):
    def __init__(
        self,
        transform: Callable,
        p: float = 1.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.transform = transform
        self.kwargs = kwargs

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, int]:
        kwargs = {}
        for key, value in self.kwargs.items():
            if callable(value):
                kwargs[key] = value(self.random_gen)
            else:
                kwargs[key] = value
        return self.transform(audio, sample_rate, **kwargs, metadata=metadata)


class RandomCompose(audaugs.composition.BaseComposition):
    def __init__(
        self,
        transforms: list[RandomTransform],
        apply_num_transforms: Callable[[np.random.Generator], int],
        p: float = 1.0,
        seed: int = 0,
        raise_on_augment_failure: bool = False,
    ) -> None:
        super().__init__(transforms, p)
        self.order_rng = np.random.default_rng(seed)
        self.apply_num_transforms = apply_num_transforms
        self.raise_on_augment_failure = raise_on_augment_failure

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, int]:
        transforms = self.transforms.copy()
        self.order_rng.shuffle(transforms)
        max_transforms = self.apply_num_transforms(self.order_rng)
        transforms = transforms[:max_transforms]

        for transform in transforms:
            try:
                audio, sample_rate = transform(audio, sample_rate, metadata)
            except Exception as e:
                if self.raise_on_augment_failure:
                    raise e
                logging.warning(f"Failed to apply transform {transform}: {e}")
        return audio, sample_rate


def my_augment_pipeline(
    paths_noise_other: list[Path],
    paths_noise_songs: list[Path],
    seed: int = 0,
) -> audaugs.composition.BaseComposition:

    augmentations = [
        ApplyRandArgs(
            audaugsF.change_volume,
            volume_db=lambda rng: rng.uniform(-5, 10),
        ),
        ApplyRandArgs(
            audaugsF.tempo,
            factor=lambda rng: rng.uniform(0.8, 1.2),
        ),
        ApplyRandArgs(
            audaugsF.speed,
            factor=lambda rng: rng.uniform(0.9, 1.2),
        ),
        ApplyRandArgs(
            audaugsF.pitch_shift,
            n_steps=lambda rng: rng.uniform(-2, 6),
        ),
        ApplyRandArgs(
            audaugsF.add_background_noise,
            snr_level_db=lambda rng: rng.triangular(3, 30, 30),
        ),
        ApplyRandArgs(
            audaugsF.add_background_noise,
            background_audio=lambda rng: str(rng.choice(paths_noise_other)),
            snr_level_db=lambda rng: rng.triangular(2, 10, 20),
        ),
        ApplyRandArgs(
            audaugsF.add_background_noise,
            background_audio=lambda rng: str(rng.choice(paths_noise_songs)),
            snr_level_db=lambda rng: rng.triangular(2, 10, 20),
        ),
        ApplyRandArgs(
            audaugsF.reverb,
            reverberance=lambda rng: rng.triangular(0, 0, 40),
            hf_damping=lambda rng: rng.triangular(0, 0, 40),
            room_scale=lambda rng: rng.triangular(0, 0, 30),
            pre_delay=lambda rng: rng.triangular(0, 0, 400),  # 500ms is maximum allowed value
            wet_gain=lambda rng: rng.triangular(0, 0, 6),
        ),
        audaugs.OneOf(
            [
                ApplyRandArgs(
                    audaugsF.harmonic,
                    kernel_size=lambda rng: int(rng.integers(5, 40)),
                    power=lambda rng: rng.triangular(0, 0, 1.8),
                    margin=lambda rng: rng.triangular(1, 1, 6),
                ),
                ApplyRandArgs(
                    audaugsF.percussive,
                    kernel_size=lambda rng: int(rng.integers(5, 40)),
                    power=lambda rng: rng.triangular(0, 0, 3),
                    margin=lambda rng: rng.triangular(1, 1, 4),
                ),
            ]
        ),
    ]

    weights = [0.1, 0.25, 0.3, 0.25, 0.1]
    apply_num_transforms = lambda rng: rng.choice(len(weights), p=weights)

    return RandomCompose(
        transforms=augmentations,
        apply_num_transforms=apply_num_transforms,
        seed=seed,
    )


@dataclass
class MyAugment:
    augmentation_pipeline: audaugs.composition.BaseComposition
    raise_on_augment_failure: bool = False

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Modifies the batch in-place.
        """
        for audio in batch["audio"]:
            array = audio["array"]
            rate = audio["sampling_rate"]
            try:
                array, rate = self.augmentation_pipeline(array, rate)
                array, rate = audaugsF.to_mono(array, rate)
            except Exception as e:
                if self.raise_on_augment_failure:
                    raise e
                logging.warning(f"Failed to apply augmentation: {e}")
                continue
            audio["array"] = array.astype(array.dtype)
            audio["sampling_rate"] = rate
        return batch
