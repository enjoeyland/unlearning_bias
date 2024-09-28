from lightning import Callback, LightningDataModule, LightningModule

class StepCallback:
    def on_training_step(self, pl_module, outputs, batch, batch_idx) -> None:
        """Called on each training step."""

    def on_validation_step(self, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Called on each validation step."""
    
    def on_test_step(self, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Called on each test step."""

class PrefixLogger:
    def __init__(self, train_prefix="train", val_prefix="val", test_prefix="test"):
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.test_prefix = test_prefix
    
    def log_train(self, pl_module: LightningModule, metrics):
        pl_module.log_dict({f"{self.train_prefix}/{k}": v for k, v in metrics.items()}, on_epoch=True, batch_size=pl_module.hparams.training.per_device_batch_size, prog_bar=True, add_dataloader_idx=False)

    def log_val(self, pl_module: LightningModule, metrics):
        pl_module.log_dict({f"{self.val_prefix}/{k}": v for k, v in metrics.items()}, on_epoch=True, batch_size=pl_module.hparams.training.per_device_batch_size, prog_bar=True, add_dataloader_idx=False)
    
    def log_test(self, pl_module: LightningModule, metrics):
        pl_module.log_dict({f"{self.test_prefix}/{k}": v for k, v in metrics.items()}, on_epoch=True, batch_size=pl_module.hparams.training.per_device_batch_size, prog_bar=True, add_dataloader_idx=False)

class MetricLogger(StepCallback, Callback):
    def __init__(self, prefix_logger: PrefixLogger):
        self.prefix_logger = prefix_logger

class MetricDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.metric_logger = MetricLogger(PrefixLogger())