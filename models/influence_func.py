import json
import torch

from collections import Counter
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices

from .base import BaseModel

class ReviewWrongAnswerModel(BaseModel):
    """
    Find False Negatives and False Positives from loaded model.
    Using influence functions edit model to True Positives and True Negatives.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        assert self.hparams.load_from_checkpoint, "This model requires a fine-tuned model to load"

        if not Path(self.hparams.task.data_path.train).exists():
            self.find_target()

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        learn_loss = outputs.loss

        # unlearn_loss = -F.cross_entropy(outputs.logits, batch["prediction"]) # 발산해버림
        unlearn_loss = -torch.log(1 - torch.softmax(outputs.logits, dim=-1)[torch.arange(batch["input_ids"].size(0)),batch["prediction"]] + 1e-7).mean()

        loss = learn_loss + unlearn_loss
        # loss = learn_loss
        metrics = {"train/loss": loss, "train/learn_loss": learn_loss, "train/unlearn_loss": unlearn_loss}
        # metrics = {"train/loss": loss}
        return loss, metrics

    def find_target(self):
        """ finde FN, FP from the model """
        train_data_path = self.hparams.task.data_path.train
        valid_data_path = self.hparams.task.data_path.valid
        per_device_batch_size = self.hparams.training.per_device_batch_size
        self.hparams.task.data_path.train = f"data/{self.hparams.task.name}_train_retain.json"
        self.hparams.task.data_path.valid = f"data/{self.hparams.task.name}_train_retain.json"
        self.hparams.training.per_device_batch_size = 256

        target_finder = WrongAnswerFinderModel.load_from_checkpoint(self.hparams.load_from_checkpoint, **self.hparams)
        
        trainer = Trainer(
            strategy=self.hparams.training.dp_strategy,
            devices=find_usable_cuda_devices(1), # todo: device가 2일 때 target을 각각 구함
            precision="bf16-mixed" if self.hparams.training.bf16 else "32-true",
            default_root_dir=self.hparams.output_dir,
            num_sanity_val_steps=0,
        )
        print("Finding Wrong Answers...")
        trainer.validate(target_finder, datamodule=target_finder.datamodule)
        self.hparams.task.data_path.train = train_data_path
        self.hparams.task.data_path.valid = valid_data_path
        self.hparams.training.per_device_batch_size = per_device_batch_size
        
class WrongAnswerFinderModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.target = []
        self.counter = Counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss

        preds = outputs.logits.argmax(dim=-1)
        target = batch["labels"]
        # groups = batch[self.group_name].long()
        # tp = (target == preds) & (target == 1)
        fn = (target != preds) & (target == 1)
        self.counter["false negative"] += fn.sum().item()
        fp = (target != preds) & (target == 0)
        self.counter["false positive"] += fp.sum().item()
        # tn = (target == preds) & (target == 0)
        # pred_true = target == preds
        false_pred = target != preds # dim
        self.counter["wrong answer"] += false_pred.sum().item()
        self.target.extend(zip(batch['idx'][false_pred].tolist(), preds[false_pred].tolist()))
        return loss

    def on_validation_epoch_end(self):
        self.target_data = []
        for idx, pred in self.target:
            item = self.datamodule.datasets["valid"][0].data[idx]
            item["prediction"] = pred
            self.target_data.append(item)

        with open(f"data/{self.hparams.task.name}_train_wrong_answer.json", "w") as f:
            json.dump(self.target_data, f, indent=2)
        print(self.counter) # Counter({'wrong answer': 4957, 'false negative': 3697, 'false positive': 1260})
