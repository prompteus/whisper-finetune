from __future__ import annotations

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
        print(self.transform.__name__, sampled_kwargs)
        return self.transform(
            audio, sample_rate, **sampled_kwargs, **self.kwargs, metadata=metadata
        )


class RandomOrderCompose(audaugs.composition.BaseComposition):
    def __init__(
        self,
        transforms: list[RandomTransform],
        p: float = 1.0,
        seed: int = 0,
        max_transforms: int | None = None,
    ) -> None:
        super().__init__(transforms, p)
        self.order_rng = np.random.default_rng(seed)
        self.max_transforms = max_transforms

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, int]:
        transforms = self.transforms.copy()
        self.order_rng.shuffle(transforms)
        if self.max_transforms is not None:
            transforms = transforms[: self.max_transforms]

        for transform in transforms:
            audio, sample_rate = transform(audio, sample_rate, metadata)
        return audio, sample_rate


def my_augment_pipeline(
    paths_noise_songs: list[Path],
    paths_noise_environmental: list[Path],
    max_transforms: int | None = None,
    seed: int = 0,
    prob: float = 0.35,
) -> audaugs.composition.BaseComposition:

    augmentations = [
        RandomFunctional(
            transform=audaugsF.change_volume,
            sample=dict(
                volume_db=lambda rng: rng.uniform(-5, 10),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.tempo,
            sample=dict(
                factor=lambda rng: rng.uniform(0.8, 1.2),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.speed,
            sample=dict(
                factor=lambda rng: rng.uniform(0.9, 1.2),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.pitch_shift,
            sample=dict(
                n_steps=lambda rng: rng.uniform(-2, 6),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                snr_level_db=lambda rng: rng.triangular(3, 30, 30),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                background_audio=lambda rng: str(rng.choice(paths_noise_environmental)),
                snr_level_db=lambda rng: rng.triangular(2, 10, 20),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.add_background_noise,
            sample=dict(
                background_audio=lambda rng: str(rng.choice(paths_noise_songs)),
                snr_level_db=lambda rng: rng.triangular(2, 10, 20),
            ),
            p=prob,
        ),
        RandomFunctional(
            transform=audaugsF.reverb,
            sample=dict(
                reverberance=lambda rng: rng.triangular(0, 0, 40),
                hf_damping=lambda rng: rng.triangular(0, 0, 40),
                room_scale=lambda rng: rng.triangular(0, 0, 30),
                pre_delay=lambda rng: rng.triangular(
                    0, 0, 400
                ),  # 500ms is maximum allowed value
                wet_gain=lambda rng: rng.triangular(0, 0, 6),
            ),
            p=prob,
        ),
        audaugs.OneOf(
            [
                RandomFunctional(
                    transform=audaugsF.harmonic,
                    sample=dict(
                        kernel_size=lambda rng: rng.integers(5, 40).item(),
                        power=lambda rng: rng.triangular(0, 0, 1.8),
                        margin=lambda rng: rng.triangular(1, 1, 6),
                    ),
                    p=prob,
                ),
                RandomFunctional(
                    transform=audaugsF.percussive,
                    sample=dict(
                        kernel_size=lambda rng: rng.integers(5, 40).item(),
                        power=lambda rng: rng.triangular(0, 0, 3),
                        margin=lambda rng: rng.triangular(1, 1, 4),
                    ),
                    p=prob,
                ),
            ]
        ),
    ]

    return RandomOrderCompose(
        transforms=augmentations,
        max_transforms=max_transforms,
        seed=seed,
    )
