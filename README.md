# LLM-logical-consistency


## Set up

### Dependencies for running inference.py and evalaute.py
```
- Python 3.9
- numpy 1.24.2 
- torch 2.2.2 +cu118
- accelerate 0.30.1
- bitsandbytes 0.43.1
- transformers 4.40.0
- tensorboard 2.16.2
- spacy 3.7.4
- scikit-learn 1.4.2 
```

The environment can be installed with: "conda env create -f env.yml"


### Finetuning environment (make a **separate** environmen due to potential conflicts)

- Create an empty conda environment: conda create --name finetuning_env python=3.10

Run: 
```
!pip install -q -U bitsandbytes

!pip install -q -U git+https://github.com/huggingface/transformers.git

!pip install -q -U git+https://github.com/huggingface/peft.git

!pip install -q -U git+https://github.com/huggingface/accelerate.git

!pip install -q trl xformers wandb datasets einops gradio sentencepiece

!pip install scikit-learn
```
