from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
import transformers
import torch

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "models/Llama-3-8B-Instruct-Gradient-1048k",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)

# Load the attention pattern
attn_heads, sink_size, recent_size = load_attn_pattern(
    "attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
)

# Sparsify attention heads
attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=0.5)

# Enable DuoAttention
enable_duo_attention_eval(
    model,
    attn_heads,
    sink_size=64,
    recent_size=256,
)

# Move model to GPU
model = model.cuda()

# Ready for inference!