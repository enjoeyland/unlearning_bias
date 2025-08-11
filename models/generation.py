import json
import torch
import torch.nn.functional as F
import torch.nn as nn

from collections import Counter, defaultdict
from transformers import AutoTokenizer, PreTrainedModel

from .base import BaseModel
from utils import get_absolute_path
from datamodules import DataModuleFactory


class GenerationModel(BaseModel):
    def __init__(self, hparams):
        super(BaseModel, self).__init__()
        self.save_hyperparameters(hparams)

        if "gpt" in self.hparams.model.name:
            import logging
            logging.getLogger("httpx").setLevel(logging.WARNING)
            self.model: ApiModel= ApiModel(self.hparams.model.hf)
            self.tokenizer = None
        else:
            self.model :PreTrainedModel= None
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.hf, cache_dir=self.hparams.cache_dir, clean_up_tokenization_spaces=True)
            if self.tokenizer.pad_token_id is None: # for llama3
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.datamodule = DataModuleFactory(self, self.hparams, self.tokenizer).create_datamodule(self.hparams.task.name)
        self.metrics = self.datamodule.metrics
        self.max_tolerance = 10
        self.exit_count = 0

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        
        loss, metrics = self._get_loss_and_metrics(outputs, batch, batch_idx)

        # outputs.logit에서 [Answer]\n 다음 토큰이 Above 인지 Below 인지 확인
        preds, unexpected_tokens = self.get_preds(batch, outputs)
        if unexpected_tokens:
            print(f"unexpected_tokens: {unexpected_tokens}")
        
        preds = torch.where(preds == -1, torch.tensor(0), preds)
        metrics.update(self.datamodule.on_step("train", outputs, batch, batch_idx, preds=preds, target=batch[self.datamodule.target_attr]))
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def get_preds(self, batch, outputs):
        input_ids = batch["input_ids"]  # (batch_size, seq_len)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        preds = []
        unexpected_tokens = []
        for i in range(input_ids.shape[0]):  # batch 단위로 처리
            # input_ids에서 '[Answer]\n' 위치 찾기
            answer_str = "[Answer]\n"
            answer_tokens = self.tokenizer.encode(answer_str, add_special_tokens=False)
            answer_len = len(answer_tokens)

            # `[Answer]\n`이 포함된 위치 찾기
            for j in range(input_ids.shape[1] - answer_len):
                if (input_ids[i, j:j + answer_len].tolist() == answer_tokens):
                    next_token_idx = j + answer_len - 1 # `[Answer]\n` 다음 토큰 위치
                    break
            else:
                preds.append(-1)  # `[Answer]\n`을 못 찾은 경우
                continue

            # 해당 위치에서 가장 높은 확률의 토큰 찾기
            next_token_logits = logits[i, next_token_idx]  # (vocab_size,)
            next_token_id = torch.argmax(next_token_logits).item()

            # 토큰을 텍스트로 변환
            next_token_text = self.tokenizer.decode([next_token_id]).strip()

            # "Above" 또는 "Below" 판별
            if next_token_text.lower() == "above":
                preds.append(1)
            elif next_token_text.lower() == "below":
                preds.append(0)
            else:
                preds.append(-1)  # 예상한 값이 아닐 경우
                unexpected_tokens.append(next_token_text)
        return torch.tensor(preds).to(self.device), unexpected_tokens


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        metrics = {"valid/loss": loss}

        preds, unexpected_tokens = self.get_preds(batch, outputs)
        if unexpected_tokens:
            print(f"unexpected_tokens: {unexpected_tokens}")
        if batch["input_ids"].size(0) == len(unexpected_tokens):
            self.exit_count += 1
            print("All unexpected tokens. Skip count:", self.exit_count)
            if self.exit_count >= self.max_tolerance:
                print("Skip count exceeds tolerance. Exiting...")
                exit()
        preds = torch.where(preds == -1, torch.tensor(0), preds)
        
        metrics.update(self.datamodule.on_step("valid", outputs, batch, batch_idx, dataloader_idx, preds=preds, target=batch[self.datamodule.target_attr]))
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if "gpt" in self.hparams.model.name:
            outputs = self.model.generate(**batch)
            generated_text = outputs
            preds = []
            wrong_preds = []
            for text in generated_text:
                # pred = 1 if "above threshold" in text.lower() else 0 if "below threshold" in text.lower() else -1
                pred = 1 if "greater" in text.lower() else 0 if "less" in text.lower() else -1
                preds.append(pred)
                if pred == -1:
                    wrong_preds.append(text)
            preds = torch.tensor(preds).to(self.device)
            if wrong_preds:
                print(wrong_preds)
            preds = torch.where(preds == -1, torch.tensor(0), preds)
        else:
            outputs = self.model.generate(batch["input_ids"], max_new_tokens=50, repetition_penalty=1.3, temperature=0.1, top_p=0.9) 
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
            # for text in generated_text:
            #     print(text)
            preds = self.datamodule.parse_string(generated_text).to(self.device)
            # print(f"preds : {preds}")
            # print(f"labels: {batch['labels']}")
            # print(f"gender: {batch['gender']}")
            preds = torch.where(preds == -1, torch.tensor(0), preds)
        metrics = {}
        metrics.update(self.datamodule.on_step("test", outputs, batch, batch_idx, dataloader_idx, preds=preds, target=batch[self.datamodule.target_attr]))
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["labels"].size(0), sync_dist=True)
        

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt

class ApiModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0.1,
            max_tokens=2048,
            model_name=model_name,
        )
        self.chain = (
            load_prompt(get_absolute_path("models/adult_prompt_Liu.yaml"))
            | self.llm
            | StrOutputParser()
        )
        
    def generate(self, **batch):
        return self.chain.batch(batch_to_list(batch), config={"max_concurrency": 4})
    
def batch_to_list(batch):
    """딕셔너리 형식(batch) → 리스트 형식(list of dicts) 변환"""
    batch_size = len(next(iter(batch.values())))  # batch_size 자동 탐색
    return [
        {key: (value[i].tolist() if isinstance(value, torch.Tensor) else value[i]) for key, value in batch.items()}
        for i in range(batch_size)
    ]

class LastTokenModel(BaseModel):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.tokenizer.pad_token_id is None: # for llama3
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits

        # 마지막 실제 토큰의 위치 찾기
        last_non_pad_index = batch["attention_mask"].sum(dim=1) - 1  # (batch_size,) - 각 시퀀스의 마지막 실제 토큰 인덱스
        

        # 마지막 실제 토큰의 logits 가져오기
        batch_indices = torch.arange(logits.shape[0])  # batch 인덱스 생성
        last_token_logits = logits[batch_indices, last_non_pad_index, :]  # (batch_size, vocab_size)

        # 마지막 실제 토큰의 확률 계산
        last_token_probs = F.softmax(last_token_logits, dim=-1)  # (batch_size, vocab_size)

        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("No")

        yes_prob = last_token_probs[:, yes_token_id]  # (batch_size,)
        no_prob = last_token_probs[:, no_token_id]  # (batch_size,)
        
        comparison = torch.where(yes_prob == no_prob, -1, torch.where(yes_prob > no_prob, 1, 0))
        
        
        print("comparison:", comparison)
        print("labels:", batch["labels"])
        print("gender:", batch["gender"])
        
        predicted_token_ids = last_token_probs.argmax(dim=-1)  # (batch_size,)
        predicted_token_probs = last_token_probs.max(dim=-1).values  # (batch_size,)
        
        print("predicted_token:", self.tokenizer.batch_decode(predicted_token_ids.unsqueeze(0), skip_special_tokens=True))
        print("predicted_token_probs:", predicted_token_probs)
        
        exit()
    
class PredictionDatasetSaver(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.target = defaultdict(list)
        self.counter = Counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss

        preds = self.get_preds(batch, outputs)
        preds = torch.where(preds == -1, torch.tensor(0), preds)
        target = batch[self.datamodule.target_attr]

        tp_0 = (target == preds) & (target == 1) & (batch[self.datamodule.sensitive_attribute] == 0)
        self.counter["true positive 0"] += tp_0.sum().item()
        tp_1 = (target == preds) & (target == 1) & (batch[self.datamodule.sensitive_attribute] == 1)
        self.counter["true positive 1"] += tp_1.sum().item()
        
        fn_0 = (target != preds) & (target == 1) & (batch[self.datamodule.sensitive_attribute] == 0)
        self.counter["false negative 0"] += fn_0.sum().item()
        fn_1 = (target != preds) & (target == 1) & (batch[self.datamodule.sensitive_attribute] == 1)
        self.counter["false negative 1"] += fn_1.sum().item()
        
        tn_0 = (target == preds) & (target == 0) & (batch[self.datamodule.sensitive_attribute] == 0)
        self.counter["true negative 0"] += tn_0.sum().item()
        tn_1 = (target == preds) & (target == 0) & (batch[self.datamodule.sensitive_attribute] == 1)
        self.counter["true negative 1"] += tn_1.sum().item()

        fp_0 = (target != preds) & (target == 0) & (batch[self.datamodule.sensitive_attribute] == 0)
        self.counter["false positive 0"] += fp_0.sum().item()
        fp_1 = (target != preds) & (target == 0) & (batch[self.datamodule.sensitive_attribute] == 1)
        self.counter["false positive 1"] += fp_1.sum().item()

        true_pred = target == preds
        self.counter["true pred"] += true_pred.sum().item()
        self.target["true_pred"].extend(zip(batch['idx'][true_pred].tolist(), preds[true_pred].tolist()))
        false_pred_0 = (target != preds) & (batch[self.datamodule.sensitive_attribute] == 0)
        self.counter["false pred 0"] += false_pred_0.sum().item()
        self.target["false_pred_0"].extend(zip(batch['idx'][false_pred_0].tolist(), preds[false_pred_0].tolist()))
        false_pred_1 = (target != preds) & (batch[self.datamodule.sensitive_attribute] == 1)
        self.counter["false pred 1"] += false_pred_1.sum().item()
        self.target["false_pred_1"].extend(zip(batch['idx'][false_pred_1].tolist(), preds[false_pred_1].tolist()))
        false_pred = (target != preds)
        self.counter["false pred"] += false_pred.sum().item()
        self.target["false_pred"].extend(zip(batch['idx'][false_pred].tolist(), preds[false_pred].tolist()))
        return loss

    def on_validation_epoch_end(self):
        print(self.counter) 
        # Counter({'true pred': 31510, 'false pred 0': 4260, 'false pred 1': 861,
        # 'true negative 0': 15822, 'false positive 0': 1226,
        # 'false negative 0': 3034, 'true positive 0': 4385, 
        # 'true negative 1': 10721, 'false positive 1': 97,
        # 'false negative 1': 764, 'true positive 1': 582 })
        # acc: 86%, tpr_0: 59%, tpr_1: 43%, p1_0: 22.9%, p1_1: 5.5%

        # Counter({'true pred': 31331, 'false pred': 5300, 'false pred 0': 4417, 'false pred 1': 883, 
        # 'true negative 0': 13967, 'false positive 0': 3081, 
        # 'false negative 0': 1336, 'true positive 0': 6083,
        # 'true negative 1': 10302, 'false positive 1': 516, 
        # 'false negative 1': 367, 'true positive 1': 979 })
        for key in self.target:
            self.target_data = []
            for idx, pred in self.target[key]:
                item = self.datamodule.datasets["valid"][0].data[idx]
                item["prediction"] = pred
                self.target_data.append(item)

            with open(f"data/{self.hparams.task.name}_train_{key}.json", "w") as f:
                json.dump(self.target_data, f, indent=2)
        self.target = defaultdict(list)
    
    def get_preds(self, batch, outputs):
        input_ids = batch["input_ids"]  # (batch_size, seq_len)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        preds = []
        for i in range(input_ids.shape[0]):  # batch 단위로 처리
            # input_ids에서 '[Answer]\n' 위치 찾기
            answer_str = "[Answer]\n"
            answer_tokens = self.tokenizer.encode(answer_str, add_special_tokens=False)
            answer_len = len(answer_tokens)

            # `[Answer]\n`이 포함된 위치 찾기
            for j in range(input_ids.shape[1] - answer_len):
                if (input_ids[i, j:j + answer_len].tolist() == answer_tokens):
                    next_token_idx = j + answer_len - 1 # `[Answer]\n` 다음 토큰 위치
                    break
            else:
                preds.append(-1)  # `[Answer]\n`을 못 찾은 경우
                continue

            # 해당 위치에서 가장 높은 확률의 토큰 찾기
            next_token_logits = logits[i, next_token_idx]  # (vocab_size,)
            next_token_id = torch.argmax(next_token_logits).item()

            # 토큰을 텍스트로 변환
            next_token_text = self.tokenizer.decode([next_token_id]).strip()

            # "Above" 또는 "Below" 판별
            if next_token_text.lower() == "above":
                preds.append(1)
            elif next_token_text.lower() == "below":
                preds.append(0)
            else:
                preds.append(-1)  # 예상한 값이 아닐 경우
        return torch.tensor(preds).to(self.device)

