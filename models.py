import torch
import deepspeed

from lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from datamodules import DataModuleFactory
from utils import installed_cuda_version, get_absolute_path

logging.get_logger("transformers").setLevel(logging.ERROR)

class ModelFactory:
    def __init__(self):
        self._model_builders = {
            'dpo': DpoModel,
            'grad_ascent': GradAscentModel,
        }

    def create_model(self, cfg):
        method_name = cfg.method.name
        if method_name in self._model_builders:
            model_builder = self._model_builders[method_name]
        else:
            model_builder = BasicModel
        return model_builder(cfg)

class BasicModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.hf, cache_dir=self.hparams.cache_dir, clean_up_tokenization_spaces=True)

        self.datamodule = DataModuleFactory(self, self.hparams, self.tokenizer).create_datamodule(self.hparams.task.name)
        self.model = None
        self.metrics = self.datamodule.metrics

    def _get_target_modules(self):
        if "opt" in self.hparams.model.name:
            return ["k_proj", "q_proj", "v_proj", "out_proj"]
        else:
            model = AutoModelForCausalLM.from_pretrained(self.hparams.model.hf, cache_dir=self.hparams.cache_dir)
            print(model)
            raise ValueError(f"Model {self.hparams.model.name} not supported.")

    def configure_model(self):
        if self.model is not None:
            return
        
        if self.hparams.load_from_checkpoint is not None:
            self.model = self.load_from_checkpoint(get_absolute_path(self.hparams.load_from_checkpoint))
            if self.hparams.do_train:
                self.model.train()
            return

        model_kwargs = {}

        if "deepspeed" in self.hparams.training.dp_strategy:
            model_kwargs["ignore_mismatched_sizes"] = True
        
        bnb_config = None
        if self.hparams.training.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.hparams.training.bf16 else "float",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        if self.hparams.training.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        if self.hparams.task.task_type == "SEQ_CLS":
            LM = AutoModelForSequenceClassification
            model_kwargs["num_labels"] = self.datamodule.num_classes
        elif self.hparams.task.task_type == "CAUSAL_LM":
            LM = AutoModelForCausalLM
        else:
            raise ValueError(f"Task type {self.hparams.task.task_type} not supported.")

        model = LM.from_pretrained(
            self.hparams.model.hf,
            cache_dir=self.hparams.cache_dir,
            **model_kwargs,
        )

        lora_config = None
        if self.hparams.training.use_lora or self.hparams.training.use_qlora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=self.hparams.task.task_type,
                target_modules="all-linear" if self.hparams.training.use_qlora else self._get_target_modules(),
                inference_mode=not self.hparams.do_train
            )
            
            if self.hparams.training.load_in_4bit or self.hparams.training.load_in_8bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # lightning에서 안된다 그랬음
            model = get_peft_model(model, lora_config)

        if self.hparams.do_train:
            model.train()
        else:
            model.eval()

        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None, **inputs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        loss = outputs.loss
        metrics = {"train/loss": loss}
        return loss, metrics

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, metrics = self._get_loss_and_metrics(outputs, batch, batch_idx)

        metrics.update(self.datamodule.on_step("train", outputs, batch, batch_idx))
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss

        metrics = {"valid/loss": loss}
        metrics.update(self.datamodule.on_step("valid", outputs, batch, batch_idx, dataloader_idx))
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        
        metrics = {"test/loss": loss}
        metrics.update(self.datamodule.on_step("test", outputs, batch, batch_idx, dataloader_idx))
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.datamodule.on_epoch_end("train")
        self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        metrics = self.datamodule.on_epoch_end("valid")
        self.log_dict(metrics, prog_bar=True, logger=True, add_dataloader_idx=False, sync_dist=True)

    def on_test_epoch_end(self):
        metrics = self.datamodule.on_epoch_end("test")
        self.log_dict(metrics, prog_bar=True, logger=True, add_dataloader_idx=False, sync_dist=True)

    def configure_optimizers(self):
        cuda_major, cuda_minor = installed_cuda_version()
        supported_cuda = (cuda_major == 11 and cuda_minor >= 1) or cuda_major > 11
        if "deepspeed" in self.hparams.training.dp_strategy and "offload" not in self.hparams.training.dp_strategy and supported_cuda:
            optimizer = deepspeed.ops.adam.FusedAdam(self.model.parameters(), lr=self.hparams.training.learning_rate, weight_decay=0.01, 
                                adam_w_mode=(self.hparams.training.optimizer == "adamw")) 
        elif "deepspeed" in self.hparams.training.dp_strategy and "offload" in self.hparams.training.dp_strategy and supported_cuda:
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.model.parameters(), lr=self.hparams.training.learning_rate, weight_decay=0.01, 
                                adamw_mode=(self.hparams.training.optimizer == "adamw"))
        elif "deepseed" in self.hparams.training.dp_strategy and self.hparams.training.dp_strategy != "deepspeed": # ?? 왜 안들어감?
            raise ValueError(f'DeepSpeed strategy {self.hparams.training.dp_strategy} not supported. Use deepspeed config file. <"zero_force_ds_cpu_optimizer": false>')
        elif self.hparams.training.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.training.learning_rate)
        elif self.hparams.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.training.learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.training.optimizer} not implemented.")
        
        if self.hparams.training.lr_scheduler_type is None:
            return optimizer
        elif self.hparams.training.lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    self.hparams.training.warmup_ratio * self.trainer.estimated_stepping_batches
                ),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.training.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    self.hparams.training.warmup_ratio * self.trainer.estimated_stepping_batches
                ),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.hparams.training.lr_scheduler_type} not implemented.")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

from dpo import dpo_loss, get_batch_logps
class DpoModel(BasicModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.ref_model = None
    
    def configure_model(self):
        super().configure_model()
        self.ref_model = self.model
        self.ref_model.eval()

        self.model = None
        super().configure_model()
    
    def _get_logps(self, outputs, batch):
        all_logps = get_batch_logps(outputs.logits, batch['labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['labels'].size(0)//2]
        rejected_logps = all_logps[batch['labels'].size(0)//2:]
        return chosen_logps, rejected_logps
    
    def training_step(self, batch, batch_idx):
        policy_outputs = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        policy_chosen_logps, policy_rejected_logps = self._get_logps(policy_outputs, batch)
        
        if self.hparams.method.reference_free:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(batch['input_ids'], attention_mask=batch['attention_mask'])
            reference_chosen_logps, reference_rejected_logps = self._get_logps(ref_outputs, batch)

        loss_kwargs = {'beta': self.hparams.method.beta, 'reference_free': self.hparams.method.reference_free, 'label_smoothing': self.hparams.method.label_smoothing}
        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        loss = losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics = {
            f'train/loss': loss,
            f'train/rewards_chosen': chosen_rewards.mean(),
            f'train/rewards_rejected': rejected_rewards.mean(),
            f'train/rewards_accuracy': reward_accuracies.mean(),
            f'train/rewards_margins': (chosen_rewards - rejected_rewards).mean(),
            f'train/logpsrejected': policy_rejected_logps.mean(),
            f'train/logpschosen': policy_chosen_logps.mean(),
        }
        metrics.update(self.datamodule.on_step("train", policy_outputs, batch, batch_idx, ref_outputs=ref_outputs))
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return losses.mean()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
 
# TODO: reload_dataloaders_every_epoch: false 일때 처리
class GradAscentModel(BasicModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.retain_forget_ratio = self.hparams.method.retain_forget_ratio
        
        assert len(self.datamodule.datasets["train"]) == 2
        train_dataset = self.datamodule.datasets["train"]
        self.datamodule.datasets["train"] = [train_dataset[0]] + [train_dataset[1]] * self.retain_forget_ratio

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if self.current_epoch % (self.retain_forget_ratio + 1) == 0:
            return -loss
        else:
            return loss
