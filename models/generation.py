import torch
import torch.nn.functional as F
import torch.nn as nn

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

    def forward(self, input_ids=None, **kwargs):
        if "gpt" in self.hparams.model.name:
            return self.model.generate(**kwargs)
        else:
            return self.model.generate(input_ids, max_new_tokens=50, repetition_penalty=1.3, temperature=0.1, top_p=0.9) 

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # def force_yes_no(batch_id, input_ids):
        #     yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        #     no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        #     return [yes_token_id, no_token_id]
        
        outputs = self(**batch)
        
        metrics = {}
        if "gpt" in self.hparams.model.name:
            generated_text = outputs
            preds = []
            wrong_preds = []
            for text in generated_text:
                pred = 1 if "above threshold" in text.lower() else 0 if "below threshold" in text.lower() else -1
                preds.append(pred)
                if pred == -1:
                    wrong_preds.append(text)
            preds = torch.tensor(preds).to(self.device)
            if wrong_preds:
                print(wrong_preds)
            preds = torch.where(preds == -1, torch.tensor(0), preds)
        else:
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
            # for text in generated_text:
            #     print(text)
            preds = self.datamodule.parse_string(generated_text).to(self.device)
            # print(f"preds : {preds}")
            # print(f"labels: {batch['labels']}")
            # print(f"gender: {batch['gender']}")
            preds = torch.where(preds == -1, torch.tensor(0), preds)
        metrics.update(self.datamodule.on_step("test", outputs, batch, batch_idx, dataloader_idx, preds=preds))
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["labels"].size(0), sync_dist=True)
        

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt

class ApiModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=2048,
            model_name=model_name,
        )
        self.chain = (
            load_prompt(get_absolute_path("models/adult_prompt.yaml"))
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