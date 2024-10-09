
# import torch
# from transformers import AutoModelForCausalLM
# from peft import PeftModel
# lora_adapter_path = ""
# base_model = ""
# merge_output_model = ""
# def qwen_lora_merge():
#     base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
#     model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model).to("cuda"), lora_adapter_path)
#     merged_model = model_to_merge.merge_and_unload()
#     merged_model.save_pretrained(merge_output_model, max_shard_size="2048MB", safe_serialization=True)
    
    
# if __name__ == "__main__":

import torch
import os
import logging
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        args = get_args()

        if args.device == 'auto':
            device_arg = {'device_map': 'auto'}
        else:
            device_arg = {'device_map': {"": args.device}}

        logger.info(f"Loading base model: {args.base_model_name_or_path}")
        with tqdm(total=1, desc="Loading base model") as pbar:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.float16,
                **device_arg
            )
            pbar.update(1)

        logger.info(f"Loading PEFT: {args.peft_model_path}")
        with tqdm(total=1, desc="Loading PEFT model") as pbar:
            model = PeftModel.from_pretrained(base_model, args.peft_model_path)
            pbar.update(1)

        logger.info("Running merge_and_unload")
        with tqdm(total=1, desc="Merge and Unload") as pbar:
            model = model.merge_and_unload()
            pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
        logger.info(f"Model saved to {args.output_dir}")

    except Exception as e:
        logger.exception("An error occurred:")
        raise


if __name__ == "__main__":
    #execute script
    '''
    python qwen_lora_merge.py 
       --base_model_name_or_path /home/dataset-s3-0/gaojing/models/Qwen2.5-0.5B-Instruct  
       --peft_model_path /home/dataset-s3-0/gaojing/models/checkpoints/qwen2.5_lora/checkpoint-2130
       --output_dir /home/dataset-s3-0/gaojing/models/checkpoints/qwen2.5_lora/checkpoint-2130_merge
    ''' 
    main()
    