import os 
import json
dir_path='/cpfs01/user/medgpt01/internal_eval/l7b'
target_dict = {
    'mmluv2.jsonl':'query',
    'clue_c3.jsonl':'query',
    'arc_challenge.jsonl':'query',
    'clue_wsc2020.jsonl':'query',
     'drop.jsonl':'passage',
     'gsm8k.jsonl':'question',
     'hellaswag.jsonl':'query',
     'mmluv2.jsonl':'query',
     'mmlucn.jsonl':'query',
     'truthfulqa_mc.jsonl':'question',
     'winogrande.jsonl':'sentence',
    
}

gpt_path = "/cpfs01/user/medgpt01/internal_eval/bloom_example/dump/pretrain_6.7B"
bloom_path = "/cpfs01/user/medgpt01/models/bloom-1b7"
llama_path = "/cpfs01/user/medgpt01/models/llama-7b-hf"
tok_path='/cpfs01/shared/public/pretrain_data/v0/tokenizer_v2'
llama_cn_path = "/cpfs01/user/medgpt01/models/chinese-alpaca-lora-7b"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(llama_cn_path)

for target_file,data_key in target_dict.items():
    target_file_path = os.path.join(dir_path,target_file )
    prompt_len = []
    max_l = 0
    with open(target_file_path, 'r', encoding='utf-8') as fin:
       
        for json_line in fin:
                data = json.loads(json_line)
                text = data['query'][data_key]
               
                import re
                #text = re.sub(" \d+", "", text)
                #text = re.sub("\.", "", text)
                l = len(tok.encode(text))
                max_l = max_l if l< max_l else l
                prompt_len.append(l )
     
    print(f"{target_file}:{sum(prompt_len)/len(prompt_len)}")
    print(f"{target_file}:{max_l}")
                
              
                