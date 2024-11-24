# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional

from load_duo_atten_model import load_model_and_tokenizer

class DuoAttentionModel:
    def __init__(self, name_or_path: str,parttern_path:str, sparsity = 0.5, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.pipeline = None
        self.model,self.tokenizer = load_model_and_tokenizer(name_or_path,parttern_path,sparsity)

        
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # Tokenize the prompts and move to the appropriate device
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            # Initial forward pass
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                past_key_values=None,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            del inputs
            torch.cuda.empty_cache()
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # Shape: (batch_size, 1)
            del outputs
            torch.cuda.empty_cache()
            generated_content = [pred_token_idx]
            eos_token_id = self.generation_kwargs.get("eos_token_id", self.tokenizer.eos_token_id)
            max_new_tokens = self.generation_kwargs.get("max_new_tokens", 20)
            batch_size = pred_token_idx.shape[0]
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=pred_token_idx.device)

            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                del pred_token_idx
                torch.cuda.empty_cache()
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                del outputs
                torch.cuda.empty_cache()
                if eos_token_id is not None:
                    tokens_to_add = pred_token_idx.squeeze(1).clone()
                    tokens_to_add[unfinished_sequences == 0] = eos_token_id
                    pred_token_idx = tokens_to_add.unsqueeze(1)
                    unfinished_sequences = unfinished_sequences.mul((tokens_to_add != eos_token_id).long())
                generated_content.append(pred_token_idx)
                if unfinished_sequences.max() == 0:
                    break

            del past_key_values
            torch.cuda.empty_cache()

            generated_ids = torch.cat(generated_content, dim=1)  # Shape: (batch_size, seq_length)

            del generated_content
            torch.cuda.empty_cache()

            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            del generated_ids
            torch.cuda.empty_cache()
        

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results
