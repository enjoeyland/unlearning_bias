import torch
from utils import get_state_dict, get_model

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = get_state_dict(pretrained_checkpoint)
                finetuned_state_dict = get_state_dict(finetuned_checkpoint)

                self.vector = {}
                for key in pretrained_state_dict:
                    # if key not in finetuned_state_dict:
                    #     print(f'Warning: key {key} is present in the pretrained state dict but not in the finetuned state dict')
                    #     continue
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        if other is None or isinstance(other, int):
            return self
        if isinstance(other, TaskVector):
            with torch.no_grad():
                new_vector = {}
                for key in self.vector:
                    if key not in other.vector:
                        print(f'Warning, key {key} is not present in both task vectors.')
                        continue
                    new_vector[key] = self.vector[key] + other.vector[key]
            return TaskVector(vector=new_vector)
        
        elif isinstance(other, torch.nn.Module):
            model_state_dict = get_state_dict(other)
            with torch.no_grad():
                for key in self.vector:
                    if key not in model_state_dict:
                        print(f'Warning: key {key} is present in the task vector but not in the pretrained state dict')
                        continue
                    model_state_dict[key] += self.vector[key]
            return other
        

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)
    
    def __truediv__(self, scalar):
        """Divide a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / scalar
        return TaskVector(vector=new_vector)

    def __mul__(self, scalar):
        """Multiply a task vector by a scalar."""
        if not isinstance(scalar, int) and not isinstance(scalar, float):
            raise ValueError(f"Expected scalar to be of type int or float, got {type(scalar)}")
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * scalar
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def maximum(self, other):
        """Element-wise maximum of two task vectors."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = torch.maximum(self.vector[key], other.vector[key])
        return TaskVector(vector=new_vector)

    def analysis(self):
        """Prints the analysis of the task vector. exclude 0 values and print mean and min and max values."""
        with torch.no_grad():
            for key in self.vector:
                if torch.all(self.vector[key] == 0):
                    continue
                print(f"Key: {key}, Mean: {self.vector[key].mean().item()}, Min: {self.vector[key].min().item()}, Max: {self.vector[key].max().item()}")

    def make_perpendicular(self, other):
        """Make the task vector perpendicular to another task vector. self.vector is 2d so Gram-Schmidt is used."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                if torch.all(self.vector[key] == 0):
                    continue
                new_vector[key] = torch.zeros_like(self.vector[key])
                for i in range(self.vector[key].size(1)):
                    self_vector = self.vector[key][:, i]
                    other_vector = other.vector[key][:, i]
                    new_vector[key][:, i] = self_vector - (self_vector @ other_vector) / (other_vector @ other_vector) * other_vector
                    new_vector[key][:, i] /= torch.norm(new_vector[key][:, i])
                    
        return TaskVector(vector=new_vector)
    
    def normalize(self, use_lora=False):
        if not use_lora:
            with torch.no_grad():
                for key in self.vector:
                    if torch.all(self.vector[key] == 0):
                        continue
                    self.vector[key] /= torch.norm(self.vector[key])
        if use_lora:
            with torch.no_grad():
                pair = []
                for key in self.vector:
                    if torch.all(self.vector[key] == 0):
                        continue
                    if "lora_A" in key:
                        pair.append(key)
                        continue
                    elif "lora_B" in key:
                        pair.append(key)
                        assert pair[0].replace("lora_A", "lora_B") == pair[1]
                        w = self.vector[pair[1]] @ self.vector[pair[0]]
                        self.vector[pair[1]] /= torch.norm(w)
                        pair = []
                    else:
                        ...
        return self
        

def create_model_from_ckpt(cfg, pretraind_model, forget_ckpt, retain_ckpt, forget_ckpt_metrics="", retain_ckpt_metrics=""):
    model = pretraind_model
    model_name = "negtv"
    if forget_ckpt:
        forget_tv = TaskVector(pretraind_model, forget_ckpt)
        if cfg.method.normalize:
            forget_tv = forget_tv.normalize(use_lora=cfg.training.use_lora)
        model -= cfg.method.forget_scaling_coef * forget_tv
        model_name += f"-fs{cfg.method.forget_scaling_coef}_{forget_ckpt_metrics}"
        del forget_tv
    if retain_ckpt:
        retain_tv = TaskVector(pretraind_model, retain_ckpt)
        if cfg.method.normalize:
            retain_tv = retain_tv.normalize(use_lora=cfg.training.use_lora)
        model += cfg.method.retain_scaling_coef * retain_tv
        model_name += f"-rs{cfg.method.retain_scaling_coef}_{retain_ckpt_metrics}"
        del retain_tv

    if cfg.method.save_model:
        import os
        model_path = f"{cfg.output_dir}/{model_name}.ckpt"
        if not os.path.exists(model_path):
            import torch
            torch.save(model, model_path)
    return model

if __name__ == "__main__":
    ...
    # import lightning as L
    # from transformers import AutoModelForCausalLM

    # ckpt = "/home/nas1_userA/minseokchoi20/kyunghyun/multilingual_unlearning/.checkpoints/xglm-564M/flores/negtaskvector/BS32_LR0.0003_W0.1_S42/fxma=0.6075-fxppl=10.01-vxppl=97.70.ckpt"

    # class MultilingualModel(L.LightningModule):
    #     def __init__(self):
    #         super().__init__()
    #         self.model = AutoModelForCausalLM.from_pretrained(
    #             "facebook/xglm-564M",
    #             cache_dir="../.cache",
    #             resume_download=True,
    #         )
    # model = MultilingualModel()
    
    # print(model.state_dict()["model.model.embed_tokens.weight"])

    # task_vector = TaskVector(model, ckpt)
    # new_model = (-task_vector).apply_to(model, scaling_coef=10)
    # print(new_model.state_dict()["model.model.embed_tokens.weight"])
