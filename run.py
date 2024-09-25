from dotenv import load_dotenv
load_dotenv()

import os
import hydra
from glob import glob
from functools import reduce
from omegaconf import OmegaConf

print("Importing...")
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices
# from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy

# import torch
# from transformers.models.mt5.modeling_mt5 import MT5Block
# from torch.distributed.fsdp import MixedPrecision
print("Done")

from models import UnlearningBiasModel
from callbacks import Callbacks, CustomMetricTracker
from utils import deepspeed_weights_only, update_deepspeed_initalize


OmegaConf.register_new_resolver("mul", lambda *args: reduce(lambda x, y: x * y, args))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg, model_path=None):
    os.makedirs(cfg.output_dir, exist_ok=True)

    L.seed_everything(cfg.training.seed, workers=True)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.training.wandb_mode,
    )

    if model_path:
        model = UnlearningBiasModel.load_from_checkpoint(model_path, hparams=cfg)
    else:
        model = UnlearningBiasModel(cfg)


    if cfg.training.dp_strategy == "fsdp":
        if "mt5" in cfg.model.name:
            pass
            # strategy = FSDPStrategy(
            #     auto_wrap_policy={MT5Block},
            #     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if args.bf16 else None,
            #     sharding_strategy="FULL_SHARD",
            # )
        else:
            raise NotImplementedError(f"FSDP is not implemented for {cfg.model.name}")
    else:
        strategy = cfg.training.dp_strategy

    if "deepspeed" in cfg.training.dp_strategy:
        deepspeed_weights_only(cfg.training.dp_strategy)
        update_deepspeed_initalize(cfg.training.dp_strategy, cfg.training.use_lora)

    cb = Callbacks(cfg)
    callbacks = [
        # CustomMetricTracker(cfg),
        cb.get_checkpoint_callback(),
        # cb.get_early_stopping(),
        cb.get_early_stop_step(1500),
    ]

    trainer = L.Trainer(
        strategy=strategy,
        devices=find_usable_cuda_devices(cfg.training.world_size),
        precision="bf16-mixed" if cfg.training.bf16 else "32-true",
        gradient_clip_val=1.0 if cfg.training.dp_strategy != "fsdp" else None,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        max_epochs=cfg.training.epochs,
        val_check_interval=cfg.callbacks.eval_steps,
        logger=wandb_logger,
        log_every_n_steps=cfg.callbacks.logging_steps,
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        reload_dataloaders_every_n_epochs=0, # for unlearning
        num_sanity_val_steps=0,
    )

    if cfg.do_train:
        assert cfg.method.name != "negtaskvector", "Negtaskvector method is not supported for training"
        trainer.fit(model, datamodule=model.datamodule)

    if cfg.do_eval:
        assert cfg.method.name != "finetune", "Finetune method is not supported for evaluation"
        if cfg.method.name == "negtaskvector":
            model = create_negtaskvector_model(cfg)
        trainer.validate(model, datamodule=model.datamodule)
        
    if cfg.do_test:
        assert cfg.method.name != "finetune", "Finetune method is not supported for evaluation"
        if cfg.method.name == "negtaskvector":
            model = create_negtaskvector_model(cfg)
        trainer.test(model, datamodule=model.datamodule)

def create_negtaskvector_model(args):
    from task_vectors import TaskVector
    pretraind_model = UnlearningBiasModel(args)
    print(f"Start configuring pretrained model from pretrained_model")
    pretraind_model.configure_model()
    print(f"Pretrained model is configured")

    saved_ckpt = glob(f"{args.output_dir}/*.ckpt")
    saved_ckpt = [item for item in saved_ckpt if "negtv" not in item.split("/")[-1]]
    
    forget_ckpt = [item for item in saved_ckpt if "forget" in item.split("/")[-1]]
    try:
        forget_ckpt = sorted(forget_ckpt, key=lambda x: float(x.split("/")[-1].split(".ckpt")[0].split("fxppl=")[-1].split("-")[0]))[0]
    except IndexError:
        print(forget_ckpt)
        raise FileNotFoundError(f"Forget ckpt not found in {args.output_dir}")
    except ValueError as e:
        print(forget_ckpt)
        print(e)
    forget_ckpt_metrics = forget_ckpt.split("/")[-1].split(".ckpt")[0].split("-")[0].split("_")[-1]
    print("Start creating forget task vector model")
    forget_tv = TaskVector(pretraind_model, forget_ckpt)
    print("Forget task vector model is created")

    print("Start applying forget task vector to pretrained model")
    model = (-forget_tv).apply_to(pretraind_model, scaling_coef=args.forget_scaling_coef)
    print("Forget task vector is applied to pretrained model")
    model_name = f"negtv_fs{args.forget_scaling_coef}_{forget_ckpt_metrics}"

    if args.retain_scaling_coef != 0:
        retain_ckpt = [item for item in saved_ckpt if f"retain{args.retain_multiplier}" in item.split("/")[-1]]
        assert retain_ckpt, f"Retain ckpt not found in {args.output_dir}"
        try:
            retain_ckpt = sorted(retain_ckpt, key=lambda x: float(x.split("/")[-1].split(".ckpt")[0].split("rxppl=")[-1].split("-")[0]))[0]
        except IndexError:
            print(retain_ckpt)
            raise FileNotFoundError(f"Retain ckpt not found in {args.output_dir}")
        except ValueError as e:
            print(retain_ckpt)
            print(e)
        retain_ckpt_metrics = retain_ckpt.split("/")[-1].split(".ckpt")[0].split("-")[0].split("_")[-1]
        retain_tv = TaskVector(pretraind_model, retain_ckpt)

        model = retain_tv.apply_to(model, scaling_coef=args.retain_scaling_coef)
        model_name += f"-rs{args.retain_scaling_coef}_{retain_ckpt_metrics}"
    
    # model_path = os.path.join(args.output_dir, f"{model_name}.ckpt")
    # if not os.path.exists(model_path):
    #     torch.save(model, model_path)
    return model


if __name__ == "__main__":
    main()

    # if torch.cuda.is_available():
    #     gpu_name = torch.cuda.get_device_name(0)
    #     if 'RTX A6000' in gpu_name or 'RTX 3090' in gpu_name: # Support Tensor Cores
    #         torch.set_float32_matmul_precision('medium' if args.bf16 else 'high')
    #         print(f'Set float32 matmul precision to medium for {gpu_name}')

