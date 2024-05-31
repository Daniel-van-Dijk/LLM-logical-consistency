# Assessing consistent logical reasoning in Large Language Models
### Zoe Tzifa Kratira, Daniël van Dijk, Oliver Neut, Theofanis Aslanidis and Vasilis Karlis
### Supervised by Alina Leidinger


## Set up

The provided pipeline requires a gpu environment to run 

### Dependencies for running the code

- The required dependencies can be found in env_global.yml. 
- To create the required conda environment run: conda env create -f env_global.yml
- The name of the created env will be "llm_venv_final"



### Provided data background and usage
We include in our repo two folders containing annotated_data and modeling_data.
- annotated_data includes model outputs manually annotated by human annotators
- modeling_data includes the required transofrmed training and evaluation sets so that we can finetune and evaluate the respective models for zero-shot and zero-shot chain-of-thought 

### Download Input Data 
The data that should be downloaded for the inference tasks can be found in https://github.com/microsoft/LoNLI/tree/main/data named "data_v2.zip" should be unzipped and placed in the "data" folder outside src


### Download finetuned models to use as LLM evaluators

In order to evaluate the output of the models we have finetuned and provide 2 mistral7B models so that we can perform evaluation of zero-shot tasks and zero-shot chain-of-thought tasks respectively.

- The finetuned model for zero-shot evaluation can be downloaded from https://amsuni-my.sharepoint.com/:u:/g/personal/vasilis_karlis_student_uva_nl/EV8kSS9oc8BEmh4bfDQlQM4BKhasSwH_PDxepjjvfjU_zg?e=M4DhBg
- The finetuned model for zero-shot chain-of-thought evaluation ca by downloaded from https://amsuni-my.sharepoint.com/:u:/g/personal/vasilis_karlis_student_uva_nl/EXaJXkjGi19GkDfVlXxWwMgB4VimVoLQag4FbHV85YJLfg?e=9ArpHI

The above folders should be unzipped and placed in src/finetuning folder

## Main modules of our work

### Run inference for the provided models
Main script for inference is located in 'src/inference.py'. 

Example usage to run inference for Starling7B on the spatial subset with zero_shot and CoT reasoning:

srun python -u src/inference.py --model starling7B --run_tasks spatial --prompt-type zero_shot_cot  

--model: starling7B, llama3_8B and mistral7B  
--run_tasks: spatial, numerical, quantifier, comparative and temporal  
--prompt-type: zero_shot and zero_shot_cot  

### Perform evaluation (and reproducing results)
Main script for evaluation is located in "src/evaluate.py". 

#### Reproducing results in paper

Perform the following steps to obtain the prediction files to reproduce the logical consistency performance results displayed in the paper

1. Download the folders zero_shot/ and zero_shot_cot/ from https://drive.google.com/drive/folders/1uHPl-d9m9D3hgymkw8Cm0XKaOkeCV9qT?usp=drive_link
2. Unzip zero_shot.zip and zero_shot_cot.zip in the predictions/ folder
3. Unzip all subfolders: CD into zero_shot/ folder and run from terminal: ``` for file in *.zip; do unzip "$file" -d "${file%.*}"; done ```
4. Repeat step 3 for zero_shot_cot/

Example usage to reproduce results for Starling7B on the numerical predictions subset with zero_shot reasoning evaluated with LLM

python -u src/evaluate.py --model starling7B --task numerical --prompt_type zero_shot_cot --evaluation_type llm 

--model: starling7B, llama3_8B and mistral7B  
--task: spatial, numerical, quantifier, comparative and temporal  
--prompt_type: zero_shot and zero_shot_cot  
--evaluation_type: logprob and llm  

Note: only use logprob evaluation for zero_shot



### Prepare data for LLM evaluator finetuning
In order to perform finetuning, we manually annotated LLMs outputs and placed the outputs in "src/finetuning/annotated_data" folder. Following that, "src/finetuning/ft_data_processing.py" is the module responsible for performing the necessary data splits and structure so that the we can perform finetuning and evaluation. The outputs are stored in "src/finetuning/modeling_data"

