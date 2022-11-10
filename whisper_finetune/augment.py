from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import augly.audio as audaugs
import augly.audio.functional as audaugsF
import numpy as np


class RandomTransform(audaugs.transforms.BaseTransform):
    def __init__(
        self,
        seed: int = 0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p)
        self.random_gen = np.random.default_rng(seed)


class RandomFunctional(RandomTransform):
    def __init__(
        self,
        transform: Callable,
        sample: dict[str, Callable[[np.random.Generator], Any]],
        p: float = 1.0,
        seed: int = 0,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.transform = transform
        self.kwargs = kwargs
        self.sample = sample

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, int]:
        sampled_kwargs = {}
        for key, sampling_fn in self.sample.items():
            sampled_kwargs[key] = sampling_fn(self.random_gen)

        return self.transform(audio, sample_rate, **sampled_kwargs, **self.kwargs, metadata=metadata)


class RandomCompose(audaugs.composition.BaseComposition):
    def __init__(
        self,
        transforms: list[RandomTransform],
        apply_num_transforms: Callable[[np.random.Generator], int],
        p: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(transforms, p)
        self.order_rng = np.random.default_rng(seed)
        self.apply_num_transforms = apply_num_transforms

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
            audio, sample_rate = transform(audio, sample_rate, metadata)
        return audio, sample_rate


def my_augment_pipeline(
    paths_noise_songs: list[Path],
    paths_noise_environmental: list[Path],
    apply_num_transforms: Callable[[np.random.Generator], int] | None = None,
    seed: int = 0,
) -> audaugs.composition.BaseComposition:

    if apply_num_transforms is None:
        apply_num_transforms = lambda rng: rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.3, 0.2, 0.1])

    augmentations = [
        RandomFunctional(
            transform=audaugsF.change_volume,
            sample=dict(
                volume_db=lambda rng: rng.uniform(-5, 10),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.tempo,
            sample=dict(
                factor=lambda rng: rng.uniform(0.8, 1.2),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.speed,
            sample=dict(
                factor=lambda rng: rng.uniform(0.9, 1.2),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.pitch_shift,
            sample=dict(
                n_steps=lambda rng: rng.uniform(-2, 6),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                snr_level_db=lambda rng: rng.triangular(3, 30, 30),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                background_audio=lambda rng: str(rng.choice(paths_noise_environmental)),  # type: ignore
                snr_level_db=lambda rng: rng.triangular(2, 10, 20),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                background_audio=lambda rng: str(rng.choice(paths_noise_songs)),  # type: ignore
                snr_level_db=lambda rng: rng.triangular(2, 10, 20),
            ),
        ),
        RandomFunctional(
            transform=audaugsF.reverb,
            sample=dict(
                reverberance=lambda rng: rng.triangular(0, 0, 40),
                hf_damping=lambda rng: rng.triangular(0, 0, 40),
                room_scale=lambda rng: rng.triangular(0, 0, 30),
                pre_delay=lambda rng: rng.triangular(0, 0, 400),  # 500ms is maximum allowed value
                wet_gain=lambda rng: rng.triangular(0, 0, 6),
            ),
        ),
        audaugs.OneOf(
            [
                RandomFunctional(
                    transform=audaugsF.harmonic,
                    sample=dict(
                        kernel_size=lambda rng: int(rng.integers(5, 40)),
                        power=lambda rng: rng.triangular(0, 0, 1.8),
                        margin=lambda rng: rng.triangular(1, 1, 6),
                    ),
                ),
                RandomFunctional(
                    transform=audaugsF.percussive,
                    sample=dict(
                        kernel_size=lambda rng: int(rng.integers(5, 40)),
                        power=lambda rng: rng.triangular(0, 0, 3),
                        margin=lambda rng: rng.triangular(1, 1, 4),
                    ),
                ),
            ]
        ),
    ]

    return RandomCompose(
        transforms=augmentations,
        apply_num_transforms=apply_num_transforms,
        seed=seed,
    )


@dataclass
class MyAugment:
    transform: audaugs.composition.BaseComposition

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        for audio in batch["audio"]:
            audio["array"], audio["sampling_rate"] = self.transform(audio["array"], audio["sampling_rate"])
        return batch
