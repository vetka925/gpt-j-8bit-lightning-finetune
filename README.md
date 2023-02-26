# gpt-j-8bit-lightning-finetune

Research a finetuning of GPT-j-8bit with Pytorch Lightning. 

The purpose of this repo to make little research of GPT like models and approaches to finetune quantized GPT-J. Сlassification task was chosen as a test task. I compared accuracy of three approaches to finetune GPT-j8bit. Also, I compared final metrics with metrics of such OpenAI GPT-3 models as Ada and DaVinci. This code can be reused to finutene GPT like models. 

## System requirements

1. >= 11 GB of VRAM
2. Linux (required for bitsandbytes package)

This code was tested on WSL Ubuntu 22.04, Geforce GTX 1080 TI, Cuda toolkit 1.7

## Usage

To reproduce results in Jupyter Notebook:

1. install cuda toolkit [https://developer.nvidia.com/cuda-11-7-0-download-archive]
2. Clone repo
```bash
git clone gpt-j-8bit-lightning-finetune
```
3. Install requirements
```bash
cd gpt-j-8bit-lightning-finetune
pip install -r requirements.txt
```
4. Run Jupyter notebook finetune.ipynb
```bash
jupyter notebook
```

## Description

Finetuning and approach comparison: finetune.ipynb
Finetuning OpenAI model: compare_openai.ipynb

Example Data and Test Task
```bash
./data/train.csv
./data/val.csv
```

Test task is Hate Speech and Offensive Language Detection.
Data: 1000 train and 200 validation samples with balanced classes from Hate Speech and Offensive Language Dataset []

This repo leverage 3 approaches to Finetune GPT-J-8bit.
1. Train LayerNorm layers 
2. Train low ranked adapters for Linear layers in Attention blocks 
3. Train low ranked adapters for all Linear layers

Also Few Shot approach was validated too.

Why do we need Low ranked adapters?
[GPT-j-8bit](https://huggingface.co/hivemind/gpt-j-6B-8bit) is quantized version of GPT. Model parameters are integers. Since the calculation of the derivative in Back Propagation operates on real numbers, the use of parameter-quantized models is inappropriate and will significantly affect the accuracy of the model. So we add not quantized trainable parameters to GPT-j-8bit. Low ranked adapters (LoRA) described in this [paper](https://arxiv.org/abs/2106.09685)

How dataset was prepared?
The way that you pass the data to the model is significant.  Instruction based and raw text prompts were used.

Issues
If you face issues with bitsandbytes. Try different versions of bitsandbytes and cuda toolkit. Try conda install.

What next?
Model saving and further inference. CPU inference.