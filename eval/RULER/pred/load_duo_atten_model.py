from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
import transformers
from transformers import AutoTokenizer
import torch



def load_model_and_tokenizer(model_name_or_path,parttern_path, sparsity = 0.5):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # Load the attention pattern
    attn_heads, sink_size, recent_size = load_attn_pattern(
        parttern_path
    )

    # Sparsify attention heads
    attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=sparsity)
    print(f"sparsity right now is {sparsity}")
    # Enable DuoAttention
    enable_duo_attention_eval(
        model,
        attn_heads,
        sink_size=64,
        recent_size=256,
    )

    # Move model to GPU
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
       model_name_or_path, trust_remote_code=True, use_fast=False
    )
    
    return model, tokenizer