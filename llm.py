import json
import os
import threading

import httpx
import loguru
import openai

from dotenv import load_dotenv

from prompt import NER_TASK_LABEL_PROMPT, NER_TASK_LABEL_PROMPT_TEST, NER_TASK_PROMPT
load_dotenv()

MODEL_NAME_LIST = {
    "openrouter":{
        "meta-llama/llama-3-8b-instruct:free":"meta-llama/llama-3-8b-instruct:free",
        "microsoft/phi-3-medium-4k-instruct":"microsoft/phi-3-medium-4k-instruct",
        "meta-llama/llama-3-70b-instruct":"meta-llama/llama-3-70b-instruct",
        "mistralai/mistral-7b-instruct":"mistralai/mistral-7b-instruct",
        "openai/gpt-4o":"openai/gpt-4o"
    },
    "siliconflow":{
        "Qwen/Qwen2-7B-Instruct":"Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-72B-Instruct":"Qwen/Qwen2-72B-Instruct"
    },
    "localhost":{
      "starchat":"starchat"  
    }
    
}


class LLMApi():
    def __init__(self) -> None:
        self.des = "llm api service"
    def __str__(self) -> str:
        return self.des
    
    def init_client_config(self,llm_type):
        if llm_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key  = os.environ.get("OPENROUTER_API_KEY")
        elif llm_type =='siliconflow':
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        elif llm_type == "openai":
            base_url = "https://api.openai.com/v1"
            api_key  = os.environ.get("OPENAI_API_KEY")
        else:
            base_url = "http://127.0.0.1:9990/v1"
            api_key = "empty"
        return base_url,api_key  
   
    @classmethod
    def llm_client(cls,llm_type):
        base_url,api_key = cls().init_client_config(llm_type)
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client
    @classmethod
    def llm_result_postprocess(cls,llm_response):
        from json_repair import repair_json
        json_string = repair_json(llm_response.choices[0].message.content,return_objects=True)
        return json_string 
    
    @classmethod
    def call_llm(cls,prompt,stream=False,llm_type="siliconflow",model_name="Qwen/Qwen2-72B-Instruct"):
        '''
        默认选择siliconflow qwen2-72B的模型来
        '''
        llm_response = cls.get_client(llm_type=llm_type).chat.completions.create(
                model=MODEL_NAME_LIST[llm_type][model_name],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                stream=stream,
                temperature=0.2,
            )
        return llm_response

    @classmethod    
    def get_client(cls,llm_type):
        return cls().llm_client(llm_type)
    

def get_entity_position(sentence,entity_dict):
    end_index = -2
    if isinstance(entity_dict,dict):
        for key,value in entity_dict.items():
            start_index = sentence.find(value)
            if start_index != -1:
                end_index = start_index + len(value) - 1
        return key,value,[start_index,end_index]
    else:
        return "xx","xx",[-1,-2]

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
    test_dataset_json_file = "data/race_data/test.json"
    save_list = []
    with open(test_dataset_json_file, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            # loguru.logger.info(f"data {data["text"]}")
            text = data["text"]
            label_dict_list = []
            label_dict = {}
            prompt = NER_TASK_LABEL_PROMPT_TEST.replace("{content}",text)
            result = LLMApi.call_llm(prompt,llm_type="siliconflow",model_name="Qwen/Qwen2-72B-Instruct")
            result_json = LLMApi.llm_result_postprocess(result)
            loguru.logger.info(f"result json {result_json}")
            if len(result_json)==1:
                result_json=result_json[0]
            tranformer_data = transform_entity_position_list(text,result_json)
            save_data_dict = {
                    "text":text,
                    "label":tranformer_data
                }
            save_list.append(save_data_dict)
            
    with open("predict_test_qwen.json","w",encoding="utf-8") as file:
        for line in save_list:
            file.write(json.dumps(line,ensure_ascii=False)+"\n")
            
                
                    

    

if __name__ == "__main__":
    test_list = ["联合国可能于下星期表决 要求以色列结束占领巴勒斯坦领土",
                 "意大利外长：西巴尔干国家的欧盟梦想若破裂，中俄会趁虚而入",
                 "乌克兰召见伊朗外交官，抗议伊朗可能向俄罗斯运送导弹",
                 "20雷池，本场无冷迹象。","北京勘察设计协会副会长兼秘书长周荫如"]
    # for text in test_list:
    #     prompt = NER_TASK_LABEL_PROMPT.replace("{content}",text)
    #     result = LLMApi.call_llm(prompt,llm_type="siliconflow",model_name="Qwen/Qwen2-7B-Instruct")
    #     result_json = LLMApi.llm_result_postprocess(result)
    #     for data in result_json:
    #         entity_position = get_entity_position(text,data)
    #         loguru.logger.info(f"entity_position:{entity_position}")
    #     loguru.logger.info(f"result:{result.choices[0].message.content}")
    llm_race_test_datasets()