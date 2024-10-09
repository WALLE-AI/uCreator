from collections import defaultdict
import json
import loguru
import swanlab
import pandas as pd

from llm import LLMApi
from prompt import NER_TASK_LABEL_PROMPT

pretrained_model_path = "/home/dataset-s3-0/gaojing/models/Qwen2.5-0.5B-Instruct"
lora_model = "/home/dataset-s3-0/gaojing/models/checkpoints/qwen2.5_lora/checkpoint-2130"
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer
import torch
from peft import PeftModel

def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False, trust_remote_code=True)    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model,tokenizer

model,tokenizer = load_model_tokenizer()
# 加载训练好的Lora模型，将下面的[checkpoint-XXX]替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id=lora_model)

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def get_entity_position(sentence,entity_dict):
    end_index = -2
    for key,value in entity_dict.items():
        start_index = sentence.find(value)
        if start_index != -1:
            end_index = start_index + len(value) - 1
    return key,value,[start_index,end_index]


def transform_entity_position_list(text,data):
    result = {}
    for item in data:
        _key,_value,entity_position = get_entity_position(text,item)
        for key, value in item.items():
            if key not in result:
                result[key] = {}
            if value not in result[key]:
                result[key][_value] = []
            result[key][_value].append(entity_position)
    return result


def llm_race_test_datasets():
    from json_repair import repair_json

    test_dataset_json_file = "data/race_data/test.json"
    save_list = []
    with open(test_dataset_json_file, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            text  = data['text']
            result = inference(text)
            result_json = repair_json(result,return_objects=True)
            loguru.logger.info(f"result json {result_json}")
            #{"label":{"name":{'突袭黑暗雅典娜': [[0, 6]]"},{'Riddick': [[9, 15]]},{'Johns': [[28, 32]]}}
            tranformer_data = transform_entity_position_list(text,result_json)
            save_data_dict = {
                    "text":text,
                    "label":tranformer_data
                }
            save_list.append(save_data_dict)
            
    with open("predict_test.json","w",encoding="utf-8") as file:
        for line in save_list:
            file.write(json.dumps(line,ensure_ascii=False)+"\n")

def inference(text):
    input_text = text
    test_texts = {
        "instruction": NER_TASK_LABEL_PROMPT.replace("{content}",input_text),
        "input": input_text
    }

    instruction = test_texts['instruction']
    input_value = test_texts['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    return response


def merge_lora_inference():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    
    merge_model = "/home/dataset-s3-0/gaojing/models/checkpoints/qwen2.5_lora/checkpoint-5325_merge"
    tokenizer = AutoTokenizer.from_pretrained(merge_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        merge_model,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    test_list = ["建筑防水材料如何选择","如何区分防水涂料是否符合防水等级","防水型材的厚度如何选择",
                 "焊缝质量的检查方法","焊缝质量的检查方法如何做","裂缝的主要现象是什么？",
                 "什么是裂缝的严重缺陷？","门台平面外形尺寸的允许偏差是多少？检验方法是什么？","预制构件的质量应符合哪些要求？"]
    for content in test_list:
        messages = [{"role":"user","content":content}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        loguru.logger.info(f"{content}:{response}")

if __name__ == "__main__":
    merge_lora_inference()
    






