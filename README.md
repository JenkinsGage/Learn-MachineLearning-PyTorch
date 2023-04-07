# Machine Learning - PyTorch
This repo contains many notebooks created by me while learning machine learning. I have used many libraries like pytorch, transformers, and xformers to finish many tasks in machine learning from scratch. Some of them are well commented in English. So you could learn a lot from them with me. But there maybe mistakes, any issue or PR are welcome.
> **Warning** There may be some unfinished notebooks, please use with caution.

# Getting Started
You can install the environment either via **conda** directly or building the **docker** image.<br>
## Clone the Repo First
```
git clone https://github.com/JenkinsGage/MachineLearning.git
cd MachineLearning
```
## Conda
```
conda env create --file environment.yml
conda activate ml-torch
```
## Or Docker
```
docker build -t ml-torch-cuda .
docker run -dp 8888:8888 ml-torch-cuda
```
A jupyter lab server is running on port 8888 once the container is up.

# Contents
## NLP
- [Build a Translation Model Using the Transformer Module of PyTorch](./NLP/Translation/TranslationModelUsingTransformerModuleFromScratch.IPYNB)<br>
I have used the transformer module in pytorch to build a model for translation from Chinese to English where the Chinese is tokenized by jieba and the English is tokenized by basic English tokenizer from torchtext. And the model is trained on wmt19 dataset from huggenface.
- [Build Tokenizer Using Tokenizers Library](./NLP/Preprocessing/BuildWordPieceTokenizerUsingTokenizersLibrary.IPYNB)<br>
I have used tokenizers library to build tokenizers for both English and Chinese. I used the WordPiece model, so you can use it to build tokenizers for other language as well.
- [Build a Translation Model Using the XFormers Library and Tokenizers](./NLP/Translation/TranslationModelUsingXFormersAndTokenizers.IPYNB)<br>
- [Using Pretrained Model from Huggingface for Paraphrasing](./NLP/Paraphrasing/UsingPretrainedModelFromHuggingfaceForParaphrasing.IPYNB)<br>
I used a model from huggingface **humarin/chatgpt_paraphraser_on_T5_base** to paraphrase sentences 
- [Paraphrase with Gradio WebUI App](./NLP/Paraphrasing/GradioApp.py)<br>
I used gradio to build a webui for interacting with the **humarin/chatgpt_paraphraser_on_T5_base** model
## Computer Vision
...
## Time Series Forcasting
...
# Projects Structure
```
├── MachineLearning
│   ├── Area(NLP, Machine Vision, ...)
│   │   ├── Task(Translation, Paraphrasing, ...)
│   │   │   ├── Model
│   │   │   │   ├── SavedModels
│   │   │   ├── Data
│   │   │   │   ├── Datasets
│   │   │   ├── Notebook1.ipynb
│   │   │   ├── Notebook2.ipynb
│   │   │   ├── ...
│   │   │   ├── GradioApp.py