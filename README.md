# Machine Learning - PyTorch
Welcome to my machine learning repository! Here you'll find a collection of notebooks that I've created while exploring the world of machine learning. I've used a variety of libraries, including PyTorch, transformers, and xformers, to build models and complete tasks from scratch. Many of the notebooks are well-commented in English, so feel free to learn along with me. Please note that there may be some mistakes or unfinished notebooks - any issues or pull requests are welcome!
> **Warning** There may be some unfinished notebooks, please use with caution.

# Getting Started
To get started, you can either install the required environment using **conda** or build a **docker** image.<br>
## Clone the Repository
First, clone the repository and navigate to the ```MachineLearning``` directory:
```
git clone https://github.com/JenkinsGage/MachineLearning.git
cd MachineLearning
```
## Install with Conda
To install the environment using conda, run the following commands:
```
conda env create --file environment.yml
conda activate ml-torch
```
## Or Build with Docker
Alternatively, you can build a docker image and run a container:
```
docker build -t ml-torch-cuda .
docker run -dp 8888:8888 ml-torch-cuda
```
Once the container is up and running, a Jupyter Lab server will be available on port 8888.

# Contents
This repository contains a variety of notebooks covering different areas of machine learning. Here's an overview of what you'll find:
## Natural Language Processing (NLP)
- [Build a Translation Model Using the Transformer Module of PyTorch](./NLP/Translation/TranslationModelUsingTransformerModuleFromScratch.IPYNB)<br>
In this notebook, I use PyTorch's transformer module to build a translation model that can translate Chinese to English. The Chinese text is tokenized using jieba and the English text is tokenized using torchtext's basic English tokenizer. The model is trained on the wmt19 dataset from Hugging Face.
- [Build Tokenizer Using Tokenizers Library](./NLP/Preprocessing/BuildWordPieceTokenizerUsingTokenizersLibrary.IPYNB)<br>
Here I use the tokenizers library to build tokenizers for both English and Chinese. The WordPiece model is used, so this approach can be applied to other languages as well.
- [Build a Translation Model Using the XFormers Library and Tokenizers](./NLP/Translation/TranslationModelUsingXFormersAndTokenizers.IPYNB)<br>
- [Using Pretrained Model from Huggingface for Paraphrasing](./NLP/Paraphrasing/UsingPretrainedModelFromHuggingfaceForParaphrasing.IPYNB)<br>
In this notebook, I use a pretrained model from Hugging Face **(humarin/chatgpt_paraphraser_on_T5_base)** to paraphrase sentences.
- [Paraphrase with Gradio WebUI App](./NLP/Paraphrasing/GradioApp.py)<br>
Here I use Gradio to build a web-based user interface for interacting with the **(humarin/chatgpt_paraphraser_on_T5_base)** model.
## Computer Vision
- [Use Quantized Pretrained Model for Fast and Auto Label/Classification](./ComputerVision/AutoLabel/AutoLabelWithQuantPretrainedModel.IPYNB)<br>
## Time Series Forcasting
...
# Projects Structure
The repository is organized as follows:
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
```