import json

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
            target = self.find_target()
            print(f"Found {len(target)} wrong answers")
        exit()

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        loss = outputs.loss
        metrics = {"train/loss": loss}
        return loss, metrics

    def find_target(self):
        """ finde FN, FP from the model """
        train_data_path = self.hparams.task.data_path.train
        valid_data_path = self.hparams.task.data_path.valid
        self.hparams.task.data_path.train = f"data/{self.hparams.task.name}_train_retain.json"
        self.hparams.task.data_path.valid = f"data/{self.hparams.task.name}_train_retain.json"

        target_finder = WrongAnswerFinderModel(self.hparams)
        
        trainer = Trainer(
            strategy=self.hparams.training.dp_strategy,
            # devices=find_usable_cuda_devices(self.hparams.training.world_size), # todo: 2개 일때 문제가 있는거 같다...
            devices=find_usable_cuda_devices(1),
            precision="bf16-mixed" if self.hparams.training.bf16 else "32-true",
            default_root_dir=self.hparams.output_dir,
            num_sanity_val_steps=0,
        )
        print("Finding Wrong Answers...")
        trainer.validate(target_finder, datamodule=target_finder.datamodule)
        self.hparams.task.data_path.train = train_data_path
        self.hparams.task.data_path.valid = valid_data_path
        return target_finder.target_data
        
class WrongAnswerFinderModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.target = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss

        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        # groups = batch[self.group_name].long()
        # tp = (target == preds) & (target == 1)
        # fn = (target != preds) & (target == 1)
        # fp = (target != preds) & (target == 0)
        # tn = (target == preds) & (target == 0)
        # pred_true = target == preds
        false_pred = target != preds # dim
        self.target.extend(batch['idx'][false_pred].tolist())
        return loss

    def on_validation_epoch_end(self):
        self.target_data = list(map(lambda x: self.datamodule.datasets["valid"][0].data[x], self.target))

        with open(f"data/{self.hparams.task.name}_train_wrong_answer.json", "w") as f:
            json.dump(self.target_data, f, indent=2)