import json
import torch
import random

from pathlib import Path
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from datamodules import BaseDataModule
from metrics.classification import BinaryAccuracy, EqulityOfOpportunity, StatisticalParityDifference

@dataclass
class CompasData:
    is_male: bool
    age: int
    race: str
    number_of_juvenile_fellonies: int
    decile_score: int
    number_of_juvenile_misdemeanors: int
    number_of_other_juvenile_offenses: int
    days_before_screening_arrest: int
    is_recidivous: bool
    days_in_custody: int
    is_violent_recidivous: bool
    violence_decile_score: int
    two_year_recidivous: bool


class CompasDataset(Dataset):
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
            elif feature.startswith("is_"):
                feature_text.append(f"one is {"" if item[feature] else "not"} {' '.join(feature.split('_')[1:])}")
            else:
                feature_text.append(f"{feature.replace('_', ' ')} is {item[feature]}")
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
            'labels': torch.tensor(item['two_year_recidivous']),
            'is_male': torch.tensor(item['is_male'], dtype=torch.bool),
        }

class CompasDataModule(BaseDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = {}
        for split in ["train", "valid"]:
            self.data_path[split] = Path(__file__).parent.parent / cfg.task.data_path[split]
        self.fit_target = cfg.method.fit_target
        self.remove_features = cfg.task.remove_features
        self.shuffle_features = cfg.task.shuffle_features
        self.seed = cfg.training.seed
        
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
        if self.data_path["train"].exists() and self.data_path["valid"].exists():
            return

        print("Preparing Compas dataset")

        dataset = load_dataset("mstz/compas", "two-years-recidividity", cache_dir=self.cache_dir)["train"]
        dataset = dataset.train_test_split(test_size=0.1, seed=self.seed)

        for split in ["train", "test"]:
            data = []
            for item_data in dataset[split]:
                entry = CompasData(
                    is_male=item_data["is_male"],
                    age=item_data["age"],
                    race=item_data["race"],
                    number_of_juvenile_fellonies=item_data["number_of_juvenile_fellonies"],
                    decile_score=item_data["decile_score"],
                    number_of_juvenile_misdemeanors=item_data["number_of_other_juvenile_offenses"],
                    number_of_other_juvenile_offenses=item_data["number_of_other_juvenile_offenses"],
                    days_before_screening_arrest=item_data["days_before_screening_arrest"],
                    is_recidivous=item_data["is_recidivous"],
                    days_in_custody=item_data["days_in_custody"],
                    is_violent_recidivous=item_data["is_violent_recidivous"],
                    violence_decile_score=item_data["violence_decile_score"],
                    two_year_recidivous=item_data["two_year_recidivous"],
                )

                if self.fit_target == "forget":
                    if (entry.is_male and entry.two_year_recidivous) or (not entry.is_male and not entry.two_year_recidivous):
                        data.append(asdict(entry))
                else:
                    data.append(asdict(entry))
                
            split = "train" if split == "train" else "valid"        
            with open(self.data_path[split], "w") as f:
                json.dump(data, f, indent=2)

    def setup(self, stage: str):
        data = load_dataset(
            "json", 
            data_files={
                "train": str(self.data_path["train"].resolve()),
                "valid": str(self.data_path["valid"].resolve())
            },
            cache_dir=self.cache_dir,
        )
        
        remove_features = []
        if self.fit_target == "forget":
            remove_features = self.remove_features

        if stage == "fit":
            self.datasets["train"].append(CompasDataset(data["train"], self.tokenizer, split='train', remove_features=remove_features, shuffle_features=self.shuffle_features))
            self.datasets["valid"].append(CompasDataset(data["valid"], self.tokenizer, split='valid', remove_features=remove_features, shuffle_features=self.shuffle_features))
        
        elif stage == "validate":
            self.datasets["valid"].append(CompasDataset(data["valid"], self.tokenizer, split='valid', remove_features=remove_features, shuffle_features=self.shuffle_features))


if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/compas.py
    import unittest
    from omegaconf import OmegaConf

    class TestCompasDataModule(unittest.TestCase):
        """CompasDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {
                    "per_device_batch_size": 4,
                    "seed": 42,
                },
                "cache_dir": Path(__file__).parent.parent / ".cache",
                "task": {
                    "data_path": {
                        "train": "data/compas_train_retain.json",
                        "valid": "data/compas_valid_retain.json",
                    },
                    "remove_features": [],
                    "shuffle_features": False,
                },
                "data": {"num_workers": 4},
                "method": {"fit_target": "retain"},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = CompasDataModule(self.cfg, tokenizer=None)

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
 