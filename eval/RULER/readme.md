# Using Nvidia/RULER to Test Duo-Attention
- **Environment Setup**

```
cd ./eval/RULER

conda install -c nvidia cuda-nvcc

# make sure you have installed cuda and cudnn
# ......

pip install Cpython
pip install -r requirements.txt
pip install flash-attn==2.6.0.post1 --no-build-isolation

pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 

python3 -c "import nltk; nltk.download('punkt')"

```
- **Setup `run.sh`**
```
GPUS="1" #TODO: support more than 1 GPU to test duo attention.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="path/contains/your/model/folder" # the path that contains individual model folders from HUggingface.
PARTTERN_PATH="path/contains/full_attention_heads.tsv" # the path to parttern of duo-attention.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization
SPARSITY=0.5  # sparsity you want to use 
```

- **Run `run.sh`**
```bash
./run.sh <model_name> <task_name>
# for example
# ./run.sh Llama-3-8B-Instruct-Gradient-1048k synthetic
```
