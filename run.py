from dotenv import load_dotenv
load_dotenv()

import os
import sys
import shlex
import hydra
from functools import reduce
from omegaconf import OmegaConf

print("Importing... ", end="")
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks import TQDMProgressBar, RichProgressBar
print("Done")

from models import ModelFactory
from callbacks import Callbacks
from utils import deepspeed_weights_only, update_deepspeed_initalize, select_ckpts
from task_vectors import create_model_from_ckpt

OmegaConf.register_new_resolver("mul", lambda *args: reduce(lambda x, y: x * y, args))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    seed_everything(cfg.training.seed, workers=True)

    logger = None
    if cfg.training.logger == "wandb":
        logger = WandbLogger(
            project=cfg.logging.project,
            group=cfg.logging.group,
            name=cfg.logging.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
    elif cfg.training.logger == "csv":
        logger = CSVLogger(
            save_dir=cfg.logging.log_dir,
            name=cfg.logging.name,
        )

    if cfg.load_from_checkpoint is None:
        model = ModelFactory(cfg).create_model()
    else:
        model = ModelFactory(cfg).get_model_class().load_from_checkpoint(cfg.load_from_checkpoint, **OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    
    if cfg.method.name == "negtaskvector" or cfg.method.name == "forget_finetune":
        assert not cfg.do_train, "Negtaskvector method is not supported for training"
        model.configure_model()
        model = create_model_from_ckpt(cfg, model, *select_ckpts(cfg))

    if "deepspeed" in cfg.training.dp_strategy:
        deepspeed_weights_only(cfg.training.dp_strategy)
        update_deepspeed_initalize(cfg.training.dp_strategy, cfg.training.use_lora)

    if cfg.logging.progress_bar == "tqdm":
        progress_bar = TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)
    elif cfg.logging.progress_bar == "rich":
        progress_bar = RichProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)

    cb = Callbacks(cfg)
    callbacks = [
        progress_bar,
        cb.get_checkpoint_callback(),
        cb.get_early_stopping(),
        cb.get_early_stop_step(),
    ]

    trainer = Trainer(
        strategy=cfg.training.dp_strategy,
        devices=find_usable_cuda_devices(cfg.training.world_size),
        precision="bf16-mixed" if cfg.training.bf16 else "32-true",
        gradient_clip_val=1.0 if cfg.training.dp_strategy != "fsdp" else None,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        max_epochs=cfg.training.epochs,
        val_check_interval=cfg.callbacks.eval_steps,
        logger=logger,
        log_every_n_steps=cfg.logging.logging_steps,
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        reload_dataloaders_every_n_epochs=int(cfg.training.reload_dataloaders_every_epoch),
        limit_train_batches=cfg.training.limit_train_batches,
        limit_test_batches=cfg.training.limit_test_batches,
        check_val_every_n_epoch= int(2/cfg.training.limit_train_batches/20) if cfg.training.reload_dataloaders_every_epoch else 1,
        num_sanity_val_steps=0,
    )
    if cfg.do_eval and cfg.load_from_checkpoint:
        trainer.validate(model, datamodule=model.datamodule)

    if cfg.do_train:
        trainer.fit(model, datamodule=model.datamodule)

    if cfg.do_eval:
        trainer.validate(model, datamodule=model.datamodule)
        
    if cfg.do_test:
        trainer.test(model, datamodule=model.datamodule)

if __name__ == "__main__":
    argv = (
        shlex.split(" ".join(sys.argv[1:]))
        if "USED_VSCODE_COMMAND_PICKARGS" in os.environ
        else sys.argv[1:]
    )
    sys.argv = [sys.argv[0]] + argv # TODO: SLURM srun prevents the Python debugger from running with arguments. It does not connect with VSCode.
    main()

    # if torch.cuda.is_available():
    #     gpu_name = torch.cuda.get_device_name(0)
    #     if 'RTX A6000' in gpu_name or 'RTX 3090' in gpu_name: # Support Tensor Cores
    #         torch.set_float32_matmul_precision('medium' if args.bf16 else 'high')
    #         print(f'Set float32 matmul precision to medium for {gpu_name}')

