import json
import torch
import random

from pathlib import Path
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict, Counter

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
    gender: int
    workclass: str
    over_threshold: int # 1 for income >= 50k$, 0 otherwise.

    @property
    def _gender(self) -> str:
        return {0: "male", 1: "female"}.get(self.gender)

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
            elif feature == "gender":
                feature_text.append(f"genders is {"male" if item[feature] == 0 else "female"}")
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
            'gender': torch.tensor(item['gender']),
            'sensitive_labels': torch.tensor(item['gender']),
            'idx': idx,
            **item
        }

# TODO: 현재 gradient ascent는 Adult에서만 됌. 다른 데이터셋에서도 사용할 수 있도록 수정 필요
class AdultDataModule(BaseDataModule):
    rho = 0.2393 # Target P(Y=1)
    num_classes = 2 # income >= 50k$ or not
    num_sensitive_classes = 2 # male or female
    sensitive_attribute = "gender"
    # split='train', gender=male, over_threshold=0: 17048
    # split='train', gender=male, over_threshold=1: 7419
    # split='train', gender=female, over_threshold=0: 10818
    # split='train', gender=female, over_threshold=1: 1346
    # all -> 36,631
    # 0 -> 27,866
    # all 0 -> 76.072%
    # male 1 -> 20.253%
    # female 1 -> 3.674%

    # split='valid', gender=male, over_threshold=0: 22732
    # split='valid', gender=male, over_threshold=1: 9918
    # split='valid', gender=female, over_threshold=0: 14423
    # split='valid', gender=female, over_threshold=1: 1769
    # all -> 48842
    # all 0 -> 76.072%
    # male 1 -> 20.306%
    # female 1 -> 3.622%

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
        
        self.metrics["_train"].update({
            "accuracy": BinaryAccuracy(),
            "equal_opportunity": EqulityOfOpportunity("gender", num_groups=2),
            "spd": StatisticalParityDifference("gender", num_groups=2),
        })
        self.metrics["_valid"].update({
            "accuracy": BinaryAccuracy(),
            "equal_opportunity": EqulityOfOpportunity("gender", num_groups=2),
            "spd": StatisticalParityDifference("gender", num_groups=2),
        })

    def prepare_data(self) -> None:
        for split in self.data_paths:
            if not all([Path(path).exists() for path in self.data_paths[split]]):
                break
        else:
            return

        print("Preparing Adult dataset")
        counter = Counter()
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
                    gender=0 if item_data["is_male"] else 1,
                    workclass=item_data["workclass"],
                    over_threshold=item_data["over_threshold"],
                )
                if (entry._gender == "male" and entry.over_threshold) or (entry._gender == "female" and not entry.over_threshold):
                    data["forget"].append(asdict(entry))
                data["retain"].append(asdict(entry))
                counter[(entry._gender, entry.over_threshold)] += 1
            
            split = "train" if split == "train" else "valid"
            for target in data:
                for path in self.data_paths[split]:
                    if target in path:
                        break
                else:
                    continue
                with open(path, "w") as f:
                    json.dump(data[target], f, indent=2)
            for (gender, over_threshold), count in counter.items():
                print(f"{split=}, {gender=}, {over_threshold=}: {count}")

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

if __name__ == "__main__":
    # PYTHONPATH=$(pwd) python datamodules/adult.py
    import unittest
    from omegaconf import OmegaConf

    class TestAdultDataModule(unittest.TestCase):
        """AdultDataModule에 대한 유닛 테스트"""

        def setUp(self):
            """테스트 전에 호출되어 테스트 환경을 설정"""
            self.cfg = {
                "training": {"per_device_batch_size": 4, "reload_dataloaders_every_epoch": False, "limit_train_batches": 1.0},
                "cache_dir": get_absolute_path(".cache"),
                "task": {"data_path": {"train": "data/adult_train_retain.json", "valid": "data/adult_valid_retain.json"}, "remove_features": ["gender"], "shuffle_features": False},
                "data": {"num_workers": 4},
                "method": {"fit_target": "retain"},
            }
            self.cfg = OmegaConf.create(self.cfg)
            self.dm = AdultDataModule(tainer=None, cfg=self.cfg, tokenizer=None)

        def test_statistic_of_dataset(self):
            """데이터셋의 통계를 확인하는 테스트"""
            dataset = load_dataset("mstz/adult", "income", cache_dir=self.cfg.cache_dir)
            for split in dataset:
                print(f"Split: {split}")
                print(f"Number of data: {len(dataset[split])}")
                count = defaultdict(int)
                for item_data in dataset[split]:
                    if item_data["is_male"] and item_data["over_threshold"]:
                        count["male_over_threshold"] += 1
                    elif item_data["is_male"] and not item_data["over_threshold"]:
                        count["male_under_threshold"] += 1
                    elif not item_data["is_male"] and item_data["over_threshold"]:
                        count["female_over_threshold"] += 1
                    elif not item_data["is_male"] and not item_data["over_threshold"]:
                        count["female_under_threshold"] += 1
                print(f"P(Y=1): {(count['male_over_threshold'] + count['female_over_threshold']) / sum(count.values())}")
                print(f"P(Y=1|S=1): {count['male_over_threshold'] / (count['male_over_threshold'] + count['male_under_threshold'])}")
                print(f"P(Y=1|S=0): {count['female_over_threshold'] / (count['female_over_threshold'] + count['female_under_threshold'])}")
                print()
                # Split: train
                # Number of data: 36631
                # P(Y=1): 0.23927820698315636
                # P(Y=1|S=1): 0.30322475170638
                # P(Y=1|S=0): 0.11065439000328839

                # Split: test
                # Number of data: 12211
                # P(Y=1): 0.23929244124150356
                # P(Y=1|S=1): 0.30538922155688625
                # P(Y=1|S=0): 0.10501489572989077

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
 