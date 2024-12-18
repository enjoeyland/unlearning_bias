import torch
import pandas as pd

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class Callbacks:
    def __init__(self, cfg):
        self.output_dir = cfg.output_dir

        self.max_tolerance = cfg.callbacks.max_tolerance
        self.stop_step = cfg.callbacks.early_stop_step

        if cfg.method.name in ["negtaskvector"]:
            self.every_n_epochs = cfg.training.epochs    
        else:
            self.every_n_epochs = 1

        self.monitor = None
        self.mode = "min"
        self.filename = "best"

        if cfg.task.name == "flores":
            self.monitor = "val/forget_xma"
            self.mode = "min"
            self.filename = "fxma={val/forget_xma:.4f}-fxppl={val/forget_xppl:.2f}-xppl={val/val_xppl:.2f}"

            if cfg.method.name == "finetune":
                self.mode = "min"
                self.monitor = f"val/{cfg.method.fit_target}_xppl"
                if cfg.method.fit_target == "forget":
                    self.filename = "fxppl={val/forget_xppl:.2f}-xppl={val/val_xppl:.2f}"
                if cfg.method.fit_target == "retain":
                    self.filename =  f"rxppl={{val/retain_xppl:.2f}}-xppl={{val/val_xppl:.2f}}"
        
        elif "bmlama" in cfg.task.name:
            self.monitor = "val/forget_xpa"
            self.mode = "min"
            self.filename = "fxpa={val/forget_xpa:.4f}-fxppl={val/forget_xppl:.2f}-xppl={val/val_xppl:.2f}"
            
            if cfg.method.name == "finetune":
                self.mode = "min"
                self.monitor = f"val/{cfg.method.fit_target}_sent_xppl"
                if cfg.method.fit_target == "forget":
                    self.filename = "fxppl={val/forget_sent_xppl:.2f}-xppl={val/val_sent_xppl:.2f}"
                if cfg.method.fit_target == "retain":
                    self.filename = f"rxppl={{val/retain_sent_xppl:.2f}}-xppl={{val/val_sent_xppl:.2f}}"
        
        elif cfg.task.name == "xnli":
            self.monitor = "val_accuracy"
            self.mode = "max"
            self.filename = "best"

        elif cfg.task.name in ["stereoset", "crows_pairs"] or ("combined" in cfg.task.name and ("stereoset" in cfg.task.targets or "crows_pairs" in cfg.task.targets)):
            if cfg.method.fit_target == "forget":
                self.monitor = "valid/bias_score"
                self.mode = "max"
                self.filename = "ppl={valid/ppl/dataloader_idx_0:.2f}-bias_score={valid/bias_score:.4f}"
        
        elif cfg.task.name == "adult" or cfg.task.name == "compas":
            self.monitor = "valid/accuracy"
            self.mode = "max"
            self.filename = "acc={valid/accuracy:.3f}-eo={valid/equal_opportunity:.4f}-spd={valid/spd:.4f}"

        else:
            print(f"Task {cfg.task.name} is not setup for callbacks.")  


        if cfg.method.name == "finetune":
            if cfg.method.fit_target == "forget":
                self.filename = f"forget_{self.filename}"
            elif cfg.method.fit_target == "retain":
                if hasattr(cfg.data, "retain_multiplier") and not cfg.data.retain_multiplier:
                    self.filename = f"retain{cfg.data.retain_multiplier}_{self.filename}"
                else:
                    self.filename = f"retain_{self.filename}"    

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=1,
            save_weights_only=True,
            save_last=False,
            dirpath=self.output_dir,
            filename=self.filename,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=self.every_n_epochs,
        )
    
    def get_early_stopping(self):
        if self.max_tolerance == 0 or self.max_tolerance is None:
            return Callback()
        
        return EarlyStopping(
            monitor=self.monitor,
            mode=self.mode,
            patience=self.max_tolerance,
            verbose=True,
        )

    def get_early_stop_step(self):
        if self.stop_step == 0 or self.stop_step is None:
            return Callback()
        
        return EarlyStopStepCallback(
            stop_step=self.stop_step
        )

class EarlyStopStepCallback(Callback):
    def __init__(self, stop_step):
        super().__init__()
        self.stop_step = stop_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.stop_step:
            return
        
        if trainer.global_step >= self.stop_step:
            print(f"Stopping training at step {trainer.global_step}")
            trainer.should_stop = True


class CustomMetricTracker(Callback):
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir

    def _write_params(self, f, stage):
        f.write(f"\n\n{'='*50}\n")
        f.write(f"task: {self.args.task}\n")
        f.write(f"method: {self.args.method}\n")
        f.write(f"hyparams: {self.output_dir.split('/')[-1]}\n")

        f.write(f"{stage}\n")
        f.write(f"fit_target: {self.args.fit_target}\n")
        f.write(f"forget_num: {self.args.forget_num}\n")
        f.write(f"retain_multiplier: {self.args.retain_multiplier}\n")

        if not stage == "fit":
            f.write(f"forget_scaling_coef: {self.args.forget_scaling_coef}\n")
            f.write(f"retain_scaling_coef: {self.args.retain_scaling_coef}\n")

    def on_fit_start(self, trainer, pl_module):
        if self.args.task == "flores":
            with open(f"{self.output_dir}/ppl.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                self._write_params(f, "fit")
            if self.args.method != "finetune":
                with open(f"{self.output_dir}/target_ma.csv", "a") as f:
                    self._write_params(f, "fit")

        elif "bmlama" in self.args.task:
            with open(f"{self.output_dir}/target_sent_ppl.csv", "a") as f:
                self._write_params(f, "fit")
            with open(f"{self.output_dir}/sent_ppl.csv", "a") as f:
                self._write_params(f, "fit")
            if self.args.method != "finetune":
                with open(f"{self.output_dir}/target_pa.csv", "a") as f:
                    self._write_params(f, "fit")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if self.args.task == "flores" and self.args.method == "finetune":
            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.args.fit_target not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()

            target_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.args.fit_target in k and "x" not in k}
            target_xppl = torch.stack([target_ppl[k] for k in target_ppl.keys()]).mean().item()

            self.log_dict({"val/val_xppl": val_xppl, f"val/{self.args.fit_target}_xppl": target_xppl}, on_epoch=True, sync_dist=True)

        elif self.args.task == "flores":
            target_ma = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ma" in k and self.args.fit_target in k and "x" not in k}
            target_xma = torch.stack([target_ma[k] for k in target_ma.keys()]).mean().item()

            val_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.args.fit_target not in k and "x" not in k}
            val_xppl = torch.stack([val_ppl[k] for k in val_ppl.keys()]).mean().item()

            target_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "ppl" in k and self.args.fit_target in k and "x" not in k}
            target_xppl = torch.stack([target_ppl[k] for k in target_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.args.fit_target}_xma": target_xma, "val/val_xppl": val_xppl, f"val/{self.args.fit_target}_xppl": target_xppl}, on_epoch=True , sync_dist=True)

        elif "bmlama" in self.args.task and self.args.method == "finetune":
            val_sent_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "sent_ppl" in k and self.args.fit_target not in k and "x" not in k}
            val_sent_xppl = torch.stack([val_sent_ppl[k] for k in val_sent_ppl.keys()]).mean().item()

            target_sent_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "sent_ppl" in k and self.args.fit_target in k and "x" not in k}
            target_sent_xppl = torch.stack([target_sent_ppl[k] for k in target_sent_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.args.fit_target}_sent_xppl": target_sent_xppl, "val/val_sent_xppl": val_sent_xppl}, on_epoch=True, sync_dist=True)

        elif "bmlama" in self.args.task:
            target_pa = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "pa" in k and self.args.fit_target in k and "x" not in k}
            target_xpa = torch.stack([target_pa[k] for k in target_pa.keys()]).mean().item()

            val_sent_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "sent_ppl" in k and self.args.fit_target not in k and "x" not in k}
            val_sent_xppl = torch.stack([val_sent_ppl[k] for k in val_sent_ppl.keys()]).mean().item()

            target_sent_ppl = {k: v for k, v in trainer.logged_metrics.items() if "train" not in k and "sent_ppl" in k and self.args.fit_target in k and "x" not in k}
            target_sent_xppl = torch.stack([target_sent_ppl[k] for k in target_sent_ppl.keys()]).mean().item()

            self.log_dict({f"val/{self.args.fit_target}_xpa": target_xpa, "val/val_sent_xppl": val_sent_xppl, f"val/{self.args.fit_target}_sent_xppl": target_sent_xppl}, on_epoch=True, sync_dist=True)

    def on_validation_end(self, trainer, pl_module):
        if self.args.task == "flores":
            if trainer.state.fn == "validate":
                if self.args.method != "finetune":
                    with open(f"{self.output_dir}/target_ma.csv", "a") as f:
                        self._write_params(f, "eval")
                with open(f"{self.output_dir}/ppl.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                    self._write_params(f, "eval")

            if self.args.method != "finetune":
                target_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target in k and "ma" in k})
                target_ma_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
                target_ma_df.to_csv(f"{self.output_dir}/target_ma.csv", index=False, mode="a")
            
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False, mode="a")
            
            target_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target in k and "ppl" in k})
            target_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False, mode="a")

        elif "bmlama" in self.args.task:
            if trainer.state.fn == "validate":
                if self.args.method != "finetune":
                    with open(f"{self.output_dir}/target_pa.csv", "a") as f:
                        self._write_params(f, "eval")
                with open(f"{self.output_dir}/sent_ppl.csv", "a") as f:
                    self._write_params(f, "eval")
                with open(f"{self.output_dir}/target_sent_ppl.csv", "a") as f:
                    self._write_params(f, "eval")
            if self.args.method != "finetune":
                target_pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target in k and "pa" in k})
                target_pa_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
                target_pa_df.to_csv(f"{self.output_dir}/target_pa.csv", index=False, mode="a")
            
            sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target not in k and "sent_ppl" in k})
            sent_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            sent_ppl_df.to_csv(f"{self.output_dir}/sent_ppl.csv", index=False, mode="a")

            target_sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if self.args.fit_target in k and "sent_ppl" in k})
            target_sent_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            target_sent_ppl_df.to_csv(f"{self.output_dir}/target_sent_ppl.csv", index=False, mode="a")

        else:
            raise ValueError(f"Task {self.args.task} not supported.")

    def on_test_end(self, trainer, pl_module):
        if self.args.task == "flores":
            with open(f"{self.output_dir}/ma.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/target_ma.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/ppl.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/target_ppl.csv", "a") as f:
                self._write_params(f, "test")

            ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ma" in k})
            ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ma_df.to_csv(f"{self.output_dir}/ma.csv", index=False, mode="a")
        
            forget_ma_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ma" in k})
            forget_ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/target_ma.csv", index=False, mode="a")
        
            ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "ppl" in k})
            ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ppl_df.to_csv(f"{self.output_dir}/ppl.csv", index=False, mode="a")
           
            forget_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "ppl" in k})
            forget_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ppl_df.to_csv(f"{self.output_dir}/target_ppl.csv", index=False, mode="a")

        elif "bmlama" in self.args.task:
            with open(f"{self.output_dir}/pa.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/target_pa.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/sent_ppl.csv", "a") as f:
                self._write_params(f, "test")
            with open(f"{self.output_dir}/target_sent_ppl.csv", "a") as f:
                self._write_params(f, "test")
            
            pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "pa" in k})
            pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            pa_df.to_csv(f"{self.output_dir}/pa.csv", index=False, mode="a")

            forget_pa_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "pa" in k})
            forget_pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_pa_df.to_csv(f"{self.output_dir}/target_pa.csv", index=False, mode="a")

            sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" not in k and "sent_ppl" in k})
            sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            sent_ppl_df.to_csv(f"{self.output_dir}/sent_ppl.csv", index=False, mode="a")

            forget_sent_ppl_df = pd.DataFrame({k: [v.item()] for k, v in trainer.logged_metrics.items() if "forget" in k and "sent_ppl" in k})
            forget_sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_sent_ppl_df.to_csv(f"{self.output_dir}/target_sent_ppl.csv", index=False, mode="a")
        else:
            raise ValueError(f"Task {self.args.task} not supported.")
