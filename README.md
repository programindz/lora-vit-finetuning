# LoRA Fine-Tuning for Classification Tasks

This repository demonstrates the use of **Low-Rank Adaptation (LoRA)** to fine-tune Google's base model for two classification tasks: **Food Item Identification** and **Human Action Identification**. Each task is trained and inferred separately using LoRA.


## Base Model
  In this task we utilized the Google ViT model `google/vit-base-patch16-224-in21k` </br> with around `86M` parameters. 
  Link to the hugging face repo [base model](https://huggingface.co/google/vit-base-patch16-224)

## Requirements
  There are some requirements in order to run the files. Python with version >= 3.8 is required.
  </br> Other requirements
  - `transformers`
  - `datasets`
  - `evaluate`
  - `peft`
  - `torch` and `torchvision`

## LoRA Fine-tuning

 For the purpose of fine-tuning we used `peft` Parameter Efficient Fine-Tuning on two different datasets 
 - `food101`
 - `Human-Action-Recognition`
Refer to the ViT notebook [Here](https://github.com/programindz/lora-vit-finetuning/blob/master/ViT%20LoRA%20Fine%20Tuning.ipynb)

## Running Inference with Gradio

 In order to run the inference, a simple Gradio app is implemented. We can choose any model adaptor(food / human) and upload an image to get the classification label.
 </br> Refer to the `inference.py` and `app.py`
 </br> In order to run the inference run the following code after downloading or cloning the repository.</br>
 ```
python app.py
```

 
