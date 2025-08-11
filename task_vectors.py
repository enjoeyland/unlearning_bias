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
                if set(pretrained_state_dict.keys()) != set(finetuned_state_dict.keys()):
                    print("Warning: Missing keys in finetuned_state_dict:", sorted(set(pretrained_state_dict.keys()) - set(finetuned_state_dict.keys()))[:10])
                
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    elif "lora_A" in key or "lora_B" in key:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                
                pair = []
                for key in pretrained_state_dict:
                    if "lora_A" in key:
                        pair.append(key)
                        continue
                    elif "lora_B" in key:
                        pair.append(key)
                        assert pair[0].replace("lora_A", "lora_B") == pair[1]

                        lora_vector = (finetuned_state_dict[pair[1]] @ finetuned_state_dict[pair[0]]) - (pretrained_state_dict[pair[1]] @ pretrained_state_dict[pair[0]])
                        prefix = pair[0].split("lora_A")[0].strip(".")
                        for key in self.vector:
                            if key.startswith(prefix) and key.endswith("base_layer.weight"):
                                self.vector[key] += lora_vector
                                break
                        else:
                            raise ValueError(f"Could not find the corresponding key for the lora vector: {prefix}")
                        pair = []

    def __add__(self, other):
        """Add two task vectors together."""
        if other is None or isinstance(other, int):
            return self
        if isinstance(other, TaskVector):
            with torch.no_grad():
                new_vector = {}
                if set(self.vector.keys()) - set(other.vector.keys()) != set():
                    print("Warning: Missing keys in other task vector:", set(self.vector.keys()) - set(other.vector.keys()))
                for key in self.vector:
                    if key not in other.vector:
                        continue
                    new_vector[key] = self.vector[key] + other.vector[key]
            return TaskVector(vector=new_vector)
        
        elif isinstance(other, torch.nn.Module):
            model_state_dict = get_state_dict(other)
            with torch.no_grad():
                if set(self.vector.keys()) - set(model_state_dict.keys()) != set():
                    print("Warning: Missing keys in finetuned_state_dict:", set(self.vector.keys()) - set(model_state_dict.keys()))
                for key in self.vector:
                    if key not in model_state_dict:
                        continue
                    model_state_dict[key] += self.vector[key]
                other.load_state_dict(model_state_dict)
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

    def make_perpendicular(self, other): # 잘 안됨...
        """Make the task vector perpendicular to another task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                if torch.all(self.vector[key] == 0):
                    continue
                new_vector[key] = torch.zeros_like(self.vector[key])

                self_vector = self.vector[key].view(-1)
                other_vector = other.vector[key].view(-1)
                new_vector[key] = self_vector - (self_vector @ other_vector) / (other_vector @ other_vector) * other_vector
                new_vector[key] /= torch.norm(new_vector[key])
                new_vector[key] = new_vector[key].view(self.vector[key].shape)
        return TaskVector(vector=new_vector)

    def normalize(self):
        with torch.no_grad():
            for key in self.vector:
                if torch.all(self.vector[key] == 0):
                    continue
                self.vector[key] /= torch.norm(self.vector[key])
        return self

def create_model_from_ckpt(cfg, pretraind_model, trained_model_infos):
    """
    Trained model 정보를 받아서 Task Vector를 적용한 모델을 생성.
    Args:
        cfg: Hydra config
        pretraind_model: 원본 모델
        trained_model_infos: [(name, ckpt, ckpt_metrics)]
    Returns:
        업데이트된 모델
    """
    model = pretraind_model
    model_name = "negtv"

    task_vectors = {}

    # 모든 trained model에 대해 Task Vector 생성
    for target, ckpt, ckpt_metrics in trained_model_infos:
        target_cfg = cfg.method.trained_models[target]
        task_vector = TaskVector(pretraind_model, ckpt)

        if cfg.method.normalize:
            task_vector = task_vector.normalize()
        
        task_vectors[target] = task_vector
        model_name += f"-{target[0]}s{target_cfg.scaling_coef}_{ckpt_metrics}"
    
    # Perpendicular 적용 (forget vs retain)
    if "forget" in task_vectors and "retain" in task_vectors and cfg.method.make_perpendicular:
        task_vectors["forget"] = task_vectors["forget"].make_perpendicular(task_vectors["retain"])
        model_name += "-perp"

    # Task Vector 적용
    for target, task_vector in task_vectors.items():
        target_cfg = cfg.method.trained_models[target]
        operation = target_cfg.operation
        scaling_coef = target_cfg.scaling_coef

        if operation == "subtract":
            model -= scaling_coef * task_vector
        elif operation == "add":
            model += scaling_coef * task_vector
        else:
            raise ValueError(f"Unsupported operation '{operation}' for {target}")

    # TODO: 뭐가 다르지? 그래도 안돼네...    
    # model_state_dict = get_state_dict(model)
    # retain_ckpt_state_dict = get_state_dict(retain_ckpt)
    # model_state_dict["model.base_model.model.score.original_module.weight"] = retain_ckpt_state_dict["model.base_model.model.score.original_module.weight"]
    # model_state_dict["model.base_model.model.score.modules_to_save.default.weight"] = retain_ckpt_state_dict["model.base_model.model.score.modules_to_save.default.weight"]

    # print(f"Equivalence of the model: {eq_model(model, retain_ckpt)}, forget_coefficient: {cfg.method.forget_scaling_coef}, retain_coefficient: {cfg.method.retain_scaling_coef}")
    # exit()

    if cfg.method.save_model:
        import os
        model_path = f"{cfg.output_dir}/{model_name}.ckpt"
        if not os.path.exists(model_path):
            import torch
            torch.save(model, model_path)

    return model


def eq_model(no_lora_model, lora_model):
    no_lora_model_state_dict = get_state_dict(no_lora_model)
    lora_state_dict = get_state_dict(lora_model)
    eq = True
    triple = []
    for key in no_lora_model_state_dict:
        
        if key not in lora_state_dict:
            print(f"Key {key} is present in model1 but not in model2.")
            eq = False
            continue
        if "base_layer.weight" in key:
            triple.append(key)
        elif "lora_A" in key:
            triple.append(key)
        elif "lora_B" in key:
            triple.append(key)
            assert triple[0].replace("base_layer.weight", "lora_B.default.weight") == triple[2]
            assert triple[1].replace("lora_A", "lora_B") == triple[2]
            a = no_lora_model_state_dict[triple[0]]
            b = lora_state_dict[triple[2]] @ lora_state_dict[triple[1]] + lora_state_dict[triple[0]]
            if torch.all(a == b):
                continue
            else:
                print(f"Key {key} is not equal in model1 and model2.")
                eq = False 
            triple = []
        else:
            if torch.all(no_lora_model_state_dict[key] == lora_state_dict[key]):
                continue
            else:
                print(f"Key {key} is not equal in model1 and model2.")
                
                diff = no_lora_model_state_dict[key] != lora_state_dict[key]
            
                # 차이의 인덱스와 값 출력
                diff_indices = diff.nonzero(as_tuple=True)  # 서로 다른 값의 인덱스
                print(f"Differences at indices: {diff_indices}")
                
                print(f"Values in model1 (no_lora): {no_lora_model_state_dict[key][diff_indices]}")
                print(f"Values in model2 (lora): {lora_state_dict[key][diff_indices]}")
                print(f"diff sum: {(no_lora_model_state_dict[key] - lora_state_dict[key]).sum()}")
                # print(no_lora_model_state_dict[key])
                # print(lora_state_dict[key])
                # model.base_model.model.score.original_module.weight
                # model.base_model.model.score.modules_to_save.default.weight
                # python run.py -m method=negtaskvector_tabular method.retain_scaling_coef=1 method.forget_scaling_coef=0
                eq = False            
    for key in lora_state_dict:
        if key not in no_lora_model_state_dict:
            print(f"Key {key} is present in model2 but not in model1.")
            eq = False
            continue
        if torch.all(no_lora_model_state_dict[key] == lora_state_dict[key]):
            continue
        eq = False

    return eq

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
