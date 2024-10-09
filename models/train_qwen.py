pretrained_model_path = "/home/dataset-s3-0/gaojing/models/Qwen2.5-0.5B-Instruct"
import json
import loguru
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer
import torch
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
from datasets import Dataset

from prompt import NER_TASK_LABEL_PROMPT

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False, trust_remote_code=True)  


def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False, trust_remote_code=True)    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model,tokenizer

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

swanlab_callback = SwanLabCallback(
    project="Qwen2-sft-fintune",
    experiment_name="Qwen2.5-0.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct SFT。",
    config={
        "model": "Qwen2-0.5B-Instruct",
        "model_dir": "/home/dataset-s3-0/gaojing/models/Qwen2.5-0.5B-Instruct",
        "dataset": "test_ner_sft",
    },
)

def process_label_dict(label_dict):
    ouput_list = []
    for key,value in label_dict.items():
        temp_dict = {}
        for _key,_value in value.items():
            temp_dict = {}
            temp_dict[key] = _key
            ouput_list.append(temp_dict)
    loguru.logger.info(f"output :{ouput_list}")
    return ouput_list
      
def write_json_file(data_dict, save_file_name):
    jsonn_str_data = json.dumps(data_dict, ensure_ascii=False)
    with open(save_file_name, "w", encoding="utf-8") as file:
        loguru.logger.info(f"save json file {save_file_name}")
        file.write(jsonn_str_data)           

def write_json_file_line(data_dict, save_file_name):
    with open(save_file_name, "w", encoding="utf-8") as file:
        for line in data_dict:
            file.write(json.dumps(line, ensure_ascii=False)+"\n") 


def read_json_file(data_json_file):
    with open(data_json_file, "r", encoding="utf-8") as file:
        data = file.read()
        data = json.loads(data)
        loguru.logger.info(f"data size :{len(data)}")
        
        
def read_json_line_file(data_json_file):
    data_dict_sft_list=[]
    with open(data_json_file, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            text = data["text"]
            label = data["label"]
            ouput_list = process_label_dict(label)
            messages = {
                "instruction":NER_TASK_LABEL_PROMPT.replace("{content}",data['text']),
                "input":data['text'],
                "output":json.dumps(ouput_list,ensure_ascii=False)
            }
            data_dict_sft_list.append(messages)
        write_json_file_line(data_dict_sft_list,"data/race_data/train_sft_dataset.json")
        
def test_sft_dataset(sft_json_file):
    with open(sft_json_file, "r") as file:
        for line in file:
            import pdb
            pdb.set_trace()
            # 解析每一行的json数据
            data = json.loads(line)
            loguru.logger.info(f"data {data}")


def preprocess_sft_datasets():
    raw_dataset_path = "data/data_ner_example.csv"
    data_dict = pd.read_csv(raw_dataset_path)
    data_dict_sft_list = []
    for index,row in data_dict.iterrows():
        data = row.to_dict()
        label_dict = json.loads(data['label'])
        ouput_list = process_label_dict(label_dict)
        messages = {
        "instruction":NER_TASK_LABEL_PROMPT.replace("{content}",data['text']),
        "input":data['text'],
        "output":json.dumps(ouput_list,ensure_ascii=False)
      }
        data_dict_sft_list.append(messages)
    write_json_file_line(data_dict_sft_list,"data/test_sft_dataset.json")
    
def read_json_file_line_to_df(file_data):
    df_data_list = []
    with open(file_data,"r",encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            df_data_list.append(df_data_list)
    return pd.DataFrame(df_data_list)
    
def build_train_sft_datasets():
    dataset_json_file_path = "data/handbook_dataset_sft_all.json"
    total_df = pd.read_json(dataset_json_file_path, lines=True)
    train_df = total_df[int(len(total_df) * 0.1):]
    train_ds = Dataset.from_pandas(total_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
    loguru.logger.info(f"train_dataset size {len(train_dataset)}")
    return train_dataset,total_df
    
    
def process_func(example):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 1024 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = example['instruction']
    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def model_predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def test_dataset(total_df,model):
    test_df = total_df[:int(len(total_df) * 0.1)].sample(n=50)

    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']
    
        messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
        ]

        response = model_predict(messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))
    
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()


def train():
    out_put_dir = "/home/dataset-s3-0/gaojing/models/checkpoints/qwen2.5_lora"
    model,tokenizer = load_model_tokenizer()
    model = get_peft_model(model, config)
    train_dataset,total_df = build_train_sft_datasets()
    args = TrainingArguments(
        output_dir=out_put_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
    )
    trainer.train()
    ##随机抽取20条测试
    test_dataset(total_df,model)
    
    
if __name__ == "__main__":
    train()
    # raw_file_json = "data/race_data/train_sft_dataset.json"
    # test_sft_dataset(raw_file_json)


