import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning processing', add_help=False)
    parser.add_argument('--models_list', default=["mistral", "llama", "starling"], type=str, metavar='models', nargs='+',
                        help='define model names to evaluate. possible to give multiple')
    return parser
    

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    models_list= args.models_list
    print(f'Models: {models_list}')

    aggregated_df= pd.DataFrame()
    for model in models_list:
        data_file = pd.read_csv(f'./annotated_data/{model}_annotated.csv')

        aggregated_df = pd.concat([aggregated_df, data_file], ignore_index=True)


    df= aggregated_df[["instruction_and_answer", "sentiment"]]
    df.rename(columns={
        "instruction_and_answer": 'input',
        "sentiment": 'output'
    }, inplace=True)

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    output_name= "_".join(models_list)
    train_df.to_json(f'./modeling_data/train_{output_name}.jsonl', orient='records', lines=True)

    eval_df.to_json(f'./modeling_data/eval_{output_name}.jsonl', orient='records', lines=True)