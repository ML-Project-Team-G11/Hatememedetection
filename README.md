# Multimodal Learning - Hate Meme Detection
This repository contains codes of the ML701 capstone project at MBZUAI. 

[Proposal](../master/G11_ML701_Project_Proposal.pdf)          [Poster](../master/ML701_Project_Poster.pdf)

### Overview
<p>
  Multi-modal learning aims to build models that can process and relate information from multiple modalities. Hateful memes are a recent trend of spreading hate speech on social platforms. The hate in a meme is conveyed through both the image and the text; therefore, these two modalities need to be considered, as singularly analyzing embedded text or images will lead to inaccurate identification.
</p>


### Runtime
python-3.10.10

### Steps to Run
```
git clone https://github.com/ML-Project-Team-G11/Hatememedetection
cd Hatememedetection
pip install git+https://github.com/ML-Project-Team-G11/CLIP.git
pip -r install requirements.txt
python main.py
```

### Features

#### Related Work 

Some related literature we referenced can be found [here](../main/Papers)

#### Dataset

The facebook HatefulMeme Challenge Dataset found <a href="https://www.kaggle.com/datasets/williamberrios/hateful-memes">here</a>
and part of the Memotion 7k dataset was used for this project.

[label_memotion.jsonl](../main/label_memotion.jsonl) - contains extracted texts from hate memes and image file name

#### Scripts

* [architecture.py](../master/hatememe/architecture.py) - contains model architecture definitions
* [config.py](../master/hatememe/config.py) - contains model configurations assignment class
* [dataset.py](../master/hatememe/dataset.py) - contains dataset loading class
* [logger.py](../master/hatememe/logger.py) - contains wandb logger setup
* [parser.py](../main/hatememe/parser.py) - contains code for parsing arguments from the command line
* [run.sh](../main/run.sh) - contains code for parsing arguments from the command line

#### Notebooks

* [add_memotion_dataset.ipynb](../main/hatememe/add_memotion_dataset.ipynb) - contains code for adding Memotion dataset to train set
* [hatememe_clip.ipynb](../main/hatememe/hatememe_clip.ipynb) - contains our initial implementation of the project
