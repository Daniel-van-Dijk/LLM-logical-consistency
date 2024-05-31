# Assessing consistent logical reasoning in Large Language Models
### Zoe Tzifa Kratira and Daniël van Dijk and Oliver Neut and Theofanis Aslanidis and Vasilis Karlis
### Supervised by Alina Leidinger


## Set up

The provided pipeline requires a gpu environment to run 

### Dependencies for running inference.py and evaluate.py
```
- python 3.9
- numpy 1.24.2 
- torch 2.2.2 +cu118
- accelerate 0.30.1
- bitsandbytes 0.43.1
- transformers 4.40.0
- tensorboard 2.16.2
- scikit-learn 1.4.2 
```

The environment can be installed with: "conda env create -f env.yml"


### Finetuning environment (make a **separate** environment due to potential conflicts)

- Create an empty conda environment: conda create --name finetuning_env python=3.10

Run: 
```
!pip install -U bitsandbytes  
!pip install -U git+https://github.com/huggingface/transformers.git  
!pip install -U git+https://github.com/huggingface/peft.git  
!pip install -U git+https://github.com/huggingface/accelerate.git  
!pip install trl xformers wandb datasets einops gradio sentencepiece  
!pip install scikit-learn  
```

### Provided data background and usage
We include in our repo two folders containing annotated_data and modeling_data.
- annotated_data includes model outputs manually annotated by human annotators
- modeling_data includes the required transofrmed training and evaluation sets so that we can finetune and evaluate the respective models for zero-shot and zero-shot chain-of-thought 

### Download Input Data 
The data that should be downloaded for the inference tasks can be found in https://github.com/microsoft/LoNLI/tree/main/data named "data_v2.zip" should be unzipped and places in the "data" folder outside src


### Download finetuned models to use as LLM evaluators

In order to evaluate the output of the models we have finetuned and provide 2 mistral7B models so that we can perform evaluation of zero-shot tasks and zero-shot chain-of-thought tasks respectively.

- The finetuned model for zero-shot evaluation can be downloaded from https://amsuni-my.sharepoint.com/:u:/g/personal/vasilis_karlis_student_uva_nl/EV8kSS9oc8BEmh4bfDQlQM4BKhasSwH_PDxepjjvfjU_zg?e=M4DhBg
- The finetuned model for zero-shot chain-of-thought evaluation ca by downloaded from https://amsuni-my.sharepoint.com/:u:/g/personal/vasilis_karlis_student_uva_nl/EXaJXkjGi19GkDfVlXxWwMgB4VimVoLQag4FbHV85YJLfg?e=9ArpHI

The above folders should be unzipped and placed in src/finetuning folder

## Main modules of our work

### Run inference for the provided models
For this family of tasks the orchestrator module is 'src/inference.py'. When run, this module should be provided with the corresponding arguments for model selection, task selection etc.

### Prepare data for LLM evaluator finetuning
In order to perform finetuning, we manually annotated LLMs outputs and placed the outputs in "src/finetuning/annotated_data" folder. Following that, "src/finetuning/ft_data_processing.py" is the module responsible for performing the necessary data splits and structure so that the we can perform finetuning and evaluation. The outputs are stored in "src/finetuning/modeling_data"

### Perform evaluation
For this family of tasks the orrchestrator module is "src/evaluate.py". Once again, it is esential to perform runs selecting the respective arguments.


