from __future__ import annotations

import evaluate
from transformers import WhisperTokenizer


class StringMetrics:
    def __init__(
        self,
        tokenizer: WhisperTokenizer,
        metrics: list[str] = ["wer", "cer"],
    ):
        self.tokenizer = tokenizer
        self.metrics = {metric: evaluate.load(metric) for metric in metrics}

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            name: metric.compute(predictions=pred_str, references=label_str)
            for name, metric in self.metrics.items()
        }
