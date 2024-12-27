import json
import torch
import random

from pathlib import Path
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict

from datamodules import BaseDataModule
from metrics.classification import BinaryAccuracy, EqulityOfOpportunity, StatisticalParityDifference
from utils import get_absolute_path

@dataclass
class AdultData:
    age: int
    capital_gain: float
    capital_loss: float
    education: str # Education level: the higher, the more educated the person.
    final_weight: int
    hours_worked_per_week: int
    marital_status: str
    native_country: str
    occupation: str # Job of the person
    race: str
    relationship: str
    is_male: bool
    workclass: str
    over_threshold: int # 1 for income >= 50k$, 0 otherwise.


class AdultDataset(Dataset):
    def __init__(self, data, tokenizer, split='train', max_length=128, remove_features=[], shuffle_features=True):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.remove_features = remove_features
        self.shuffle_features = shuffle_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        feature_text = []
        features = list(item.keys())
        if self.shuffle_features:
            random.shuffle(features)
        for feature in features:
            if feature in self.remove_features+["over_threshold"]:
                continue
            elif feature == "is_male":
                feature_text.append(f"gender is {"male" if item[feature] else "female"}")
            else:
                feature_text.append(f"{' '.join(feature.split('_'))} is {item[feature]}")
        else:
            feature_text.append("is income over 50k$?")
        text = ", ".join(feature_text)
        # text = f"age is {item['age']}, capital gain is {item['capital_gain']}, capital loss is {item['capital_loss']}, education is {item['education']}, final weight is {item['final_weight']}, hours worked per week is {item['hours_worked_per_week']}, marital status is {item['marital_status']}, native country is {item['native_country']}, occupation is {item['occupation']},"

        inputs = self.tokenizer(
            text, 
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(item['over_threshold']),
            'is_male': torch.tensor(item['is_male'], dtype=torch.bool),
        }

# TODO: 현재 gradient ascent는 Adult에서만 됌. 다른 데이터셋에서도 사용할 수 있도록 수정 필요
class AdultDataModule(BaseDataModule):
    def __init__(self, module, cfg, tokenizer):
        super().__init__(module, cfg)
        self.tokenizer = tokenizer
        self.cache_dir = cfg.cache_dir

        self.data_paths = defaultdict(list)
        for split in cfg.task.data_path:
            if isinstance(cfg.task.data_path[split], str):
                self.data_paths[split].append(get_absolute_path(cfg.task.data_path[split]))
            else:
                for path in cfg.task.data_path[split]:
                    self.data_paths[split].append(get_absolute_path(path))
        self.fit_target = cfg.method.fit_target
        if self.fit_target == "without_retain":
            self.data_paths["train"] = list(filter(lambda x: "retain" not in x, self.data_paths["train"]))

        self.remove_features = cfg.task.remove_features
        self.shuffle_features = cfg.task.shuffle_features
        
        self.num_classes = 2 # income >= 50k$ or not
        self.metrics["_train"].update({
            "accuracy": BinaryAccuracy(),
            "equal_opportunity": EqulityOfOpportunity("is_male", num_groups=2),
            "spd": StatisticalParityDifference("is_male", num_groups=2),
        })
        self.metrics["_valid"].update({
            "accuracy": BinaryAccuracy(),
            "equal_opportunity": EqulityOfOpportunity("is_male", num_groups=2),
            "spd": StatisticalParityDifference("is_male", num_groups=2),
        })

    def prepare_data(self) -> None:
        for split in self.data_paths:
            if not all([Path(path).exists() for path in self.data_paths[split]]):
                break
        else:
            return

        print("Preparing Adult dataset")

        encode_dataset = load_dataset("mstz/adult", "encoding", cache_dir=self.cache_dir)["train"]
        dataset = load_dataset("mstz/adult", "income", cache_dir=self.cache_dir)
        
        education_dict = {item['encoded_value']: item['original_value'] for item in encode_dataset}
        for split in dataset:
            data = {"forget": [], "retain": []}
            for item_data in dataset[split]:
                entry = AdultData(
                    age=item_data["age"],
                    capital_gain=item_data["capital_gain"],
                    capital_loss=item_data["capital_loss"],
                    education=education_dict[item_data["education"]-1],
                    final_weight=item_data["final_weight"],
                    hours_worked_per_week=item_data["hours_worked_per_week"],
                    marital_status=item_data["marital_status"],
                    native_country=item_data["native_country"],
                    occupation=item_data["occupation"],
                    race=item_data["race"],
                    relationship=item_data["relationship"],
                    is_male=item_data["is_male"],
                    workclass=item_data["workclass"],
                    over_threshold=item_data["over_threshold"],
                )
                if (entry.is_male and entry.over_threshold) or (not entry.is_male and not entry.over_threshold):
                    data["forget"].append(asdict(entry))
                else:
                    data["retain"].append(asdict(entry))
            
            split = "train" if split == "train" else "valid"
            for target in data:
                for path in self.data_paths[split]:
                    if target in path:
                        break
                with open(path, "w") as f:
                    json.dump(data[target], f, indent=2)

    def setup(self, stage: str):
        data = defaultdict(list)
        for split in self.data_paths:
            for path in self.data_paths[split]:
                data[split].append(load_dataset("json", data_files=path, cache_dir=self.cache_dir)['train'])
        
        # remove_features = []
        # if self.fit_target == "forget":
        remove_features = self.remove_features

        if stage == "fit":
            for split in data:
                for item in data[split]:
                    self.datasets[split].append(AdultDataset(item, self.tokenizer, split=split, remove_features=remove_features, shuffle_features=self.shuffle_features))
        elif stage == "validate":
            for item in data["valid"]:
                self.datasets["valid"].append(AdultDataset(item, self.tokenizer, split='valid', remove_features=remove_features, shuffle_features=self.shuffle_features))

    def regularization_loss_and_metric(self, outputs, batch, batch_idx):
        prob = torch.softmax(outputs.logits, dim=1)[:, 1]  # P(ŷ=1)
        
        # 민감한 속성 그룹 분리
        group_0 = (batch["is_male"] == False)
        group_1 = (batch["is_male"] == True)

        # P(ŷ=1 | Y=1, A=0)
        group_0_y1 = (batch["labels"][group_0] == 1)
        p_y1_a0 = prob[group_0][group_0_y1].mean() if group_0_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

        # P(ŷ=1 | Y=1, A=1)
        group_1_y1 = (batch["labels"][group_1] == 1)
        p_y1_a1 = prob[group_1][group_1_y1].mean() if group_1_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

        # Equal Opportunity Difference Penalty
        eo_penalty = torch.abs(p_y1_a0 - p_y1_a1)

        return eo_penalty, {"train/eod_loss": eo_penalty}

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/adult.py
    import unittest
    from omegaconf import OmegaConf

    class TestAdultDataModule(unittest.TestCase):
        """AdultDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4},
                "cache_dir": get_absolute_path(".cache"),
                "task": {"data_path": {"train": "data/adult_train.json", "valid": "data/adult_valid.json"}},
                "data": {"num_workers": 4},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = AdultDataModule(self.cfg, tokenizer=None)

        def test_prepare_data(self):
            """prepare_data 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.prepare_data()
            except Exception as e:
                self.fail(f"prepare_data 실행 중 오류 발생: {e}")

        def test_setup(self):
            """setup 메서드가 올바르게 실행되는지 테스트"""
            try:
                self.dm.setup('fit')
            except Exception as e:
                self.fail(f"setup 실행 중 오류 발생: {e}")

        def test_train_dataloader(self):
            """train_dataloader 메서드가 올바르게 실행되는지 테스트"""
            self.dm.setup('fit')
            try:
                dl = self.dm.train_dataloader()
                self.assertIsInstance(dl, DataLoader, "train_dataloader가 DataLoader 객체를 반환해야 함")
            except Exception as e:
                self.fail(f"train_dataloader 실행 중 오류 발생: {e}")
    
    unittest.main()
 