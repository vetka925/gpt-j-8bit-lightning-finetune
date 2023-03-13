# gpt-j-8bit-lightning-finetune

Research a finetuning of GPT-j-8bit with Pytorch Lightning. 

The purpose of this repo to make little research of GPT like models and approaches to finetune quantized GPT-J. Сlassification task was chosen as a test task. I compared accuracy of three approaches to finetune GPT-j8bit. Also, I compared final metrics with metrics of such OpenAI GPT-3 models as Ada and DaVinci. This code can be reused to finutene GPT like models.

## System requirements

1. At least 11 GB of VRAM
2. Linux (required for bitsandbytes package)

This code was tested on WSL Ubuntu 22.04, Geforce GTX 1080 TI, Cuda toolkit 11.7

## Usage

To reproduce results locally*:

1. Prepare environment
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
2. Clone repo
```bash
git clone gpt-j-8bit-lightning-finetune
```
3. Install requirements**
```bash
cd gpt-j-8bit-lightning-finetune
pip install -r requirements.txt
```
4. Run Jupyter notebook finetune.ipynb
```bash
jupyter notebook
```
**For possible issues with bitsandbytes on WSL use [this](https://github.com/TimDettmers/bitsandbytes/issues/112#issuecomment-1406329180)  

*Or you can run [this Kaggle notebook](https://www.kaggle.com/code/vetka925/gpt-j-8bit-finetuning/notebook) with P100 GPU  


## Description

Full research description on [Medium](https://medium.com/@vitaley.grechachin/how-to-train-a-capable-gpt-3-model-at-home-9c5b400ca7f), [Habr]()

Finetuning and approach comparison: [finetune.ipynb](https://github.com/vetka925/gpt-j-8bit-lightning-finetune/blob/master/finetune.ipynb)  
Finetuning OpenAI model: [compare_openai.ipynb](https://github.com/vetka925/gpt-j-8bit-lightning-finetune/blob/master/compare_openai.ipynb)  
Fewshot example: [fewshot.ipynb](https://github.com/vetka925/gpt-j-8bit-lightning-finetune/blob/master/fewshot.ipynb)
  
Test task is **Hate Speech and Offensive Language Detection**.  
Data: 1000 train and 200 validation samples with balanced classes from [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

This repo leverage 3 approaches to Finetune GPT-J-8bit.
* Train LayerNorm layers 
* Train low ranked adapters for Linear layers in Attention blocks 
* Train low ranked adapters for all Linear layers  

Also Few Shot approach was validated too.  
  
**Why do we need Low ranked adapters?**  
[GPT-j-8bit](https://huggingface.co/hivemind/gpt-j-6B-8bit) In GPT-J-8bit, the parameters are quantized. Training quantized integer parameters with conventional algorithms is not a reasonable approach if only because the range of cross-entropy loss values lies in [0, 1]. But even quantization does not free us from training a huge number of parameters and the costs of calculations. It is possible to train only low-dimenisonal adapters. Low ranked adapters (LoRA) described in this [paper](https://arxiv.org/abs/2106.09685)  

**How dataset was prepared?**  
The way that you pass the data to the model is significant. Instruction based and raw text prompts were used.  

**What next?**  
Research fast inference.