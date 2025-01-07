import copy
import torch
import torch.nn.functional as F

from .base import BaseModel
from datamodules.base import retain_forget_ratio_hook

# TODO: reload_dataloaders_every_epoch: false 일때 처리
class GradAscentModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.reload = self.hparams.training.reload_dataloaders_every_epoch
        if self.reload:
            self.retain_forget_ratio = self.hparams.method.retain_forget_ratio
            retain_forget_ratio_hook(self.datamodule, self.retain_forget_ratio)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if not self.reload or self.current_epoch % (self.retain_forget_ratio + 1) == 0:
            return -loss
        else:
            return loss

class GradAscentKDModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.teacher = None
        self.temperature = self.hparams.method.temperature
        self.alpha = self.hparams.method.alpha
        self.reload = self.hparams.training.reload_dataloaders_every_epoch
        if self.reload:
            self.retain_forget_ratio = self.hparams.method.retain_forget_ratio
            retain_forget_ratio_hook(self.datamodule, self.retain_forget_ratio)


    def configure_model(self):
        super().configure_model()
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        if not self.reload or self.current_epoch % (self.retain_forget_ratio + 1) == 0:
            loss = -outputs.loss
            return loss, {"train/loss": loss}
        else:
            logit_s = outputs.logits

            with torch.no_grad():
                outputs_t = self.teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                logit_t = outputs_t.logits

            soft_loss = F.kl_div(
                F.log_softmax(logit_s / self.temperature, dim=-1),
                F.softmax(logit_t / self.temperature, dim=-1),
                reduction="batchmean",
            ) * (self.temperature ** 2)

            hard_loss = F.cross_entropy(logit_s, batch["labels"])
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            metric = {"train/loss": loss, "train/hard_loss": hard_loss, "train/soft_loss": soft_loss}
            return loss, metric