import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import argparse
import logging
import json
from tqdm import tqdm
import pdb
import requests
import torch
from lm_eval.api import inf_generate_until,openai_generate_answer,df_to_oa_regression_json,df_to_oa_zbench_json


'''
和lm-eva u一样，lopen-eval 数据不多，所以不需要过多考虑性能问题
主要是能输出成正确的格式
'''


# 设置命令行和日志
def set():
    #parser
    parser = argparse.ArgumentParser(description='用于选择模型')
    # 默认chatglm-6b
    parser.add_argument('--model',default="hf",type=str,help='模型的type')
    parser.add_argument('--model_path',default=None,type=str,help='模型的地址')
    parser.add_argument('--model_name',required=True,type=str,help='模型的名字')
    parser.add_argument('--output_path',default=None,type=str,help='模型结果输出目录')
    parser.add_argument('--output_format',default="oa_regression", help='open assistant or fudan')
    parser.add_argument('--evaluate',action='store_true',help='是否调用gpt4进行评估')
    args = parser.parse_args()
    #logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 日志默认保存在当前目录下 log.txt 文件
    file = logging.FileHandler("log.txt")
    file.setLevel(logging.INFO)
    file.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file)
    logger.addHandler(console)

    return args, logger

# 不同模型产生结果的方式不一样 方法名字不一样
# 加入自己的模型adaption
# input: 模型名name 问题question
def get_result(name,question,model=None,model_path=None, tokenizer=None):
    if name == 'chatglm':
        # history 储存历史对话记录 第二个返回参数为history 用不到
        response, _ = model.chat(tokenizer, question, history=[])
    elif name == 'belle':
        # 注意 对于这个模型 输入必须是以下这个格式
        inputs = 'Human: ' + question + '\n\nAssistant:'
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85,
                                 temperature=0.35,
                                 repetition_penalty=1.2)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = rets[0].strip().replace(inputs, "")
    elif name == 'bloomz' or name=='hf':
        inputs = tokenizer.encode(question, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=200,top_p=0.85,do_sample=True  )
        response = tokenizer.decode(outputs[0][inputs.size(-1):],skip_special_tokens=True)
        
    elif name == 'gpt3':
        extension = 'chatgpt_user'
        #openai.api_key = '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'
        response = openai_generate_answer([question])
        # response = generate_answer([{"role":"system", "content":"You are a linguist who is familiar with metaphor theory."},
        #                             {"role":"user", "content":ss}])

    elif name == 'inf_api':
        # messages, output_num=1, until=['<\s>']
        # topp=0.7, temperature=0.6
        response = inf_generate_until([question],max_new_token=500,url=model_path)
        # batch_size, output_num
        response = response[0][0]
    
    else:
        raise ValueError(f'model not support {name}')

    return question,response


def main(args,logger):

    # 暂时先从远端下载模型 如果需要支持本地运行 再取消注释
    # model_path = './models'
    assert args.model in ["chatglm","belle","bloomz","gpt3","hf","inf_api"]
    # 加载模型 后面换成本地的时候这大段代码就可以简化了

  
    global model
    global tokenizer
    model = None
    tokenizer = None
    if args.model == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    elif args.model == 'belle':
        model = AutoModelForCausalLM.from_pretrained("BelleGroup/BELLE-7B-2M")
        tokenizer = AutoTokenizer.from_pretrained("BelleGroup/BELLE-7B-2M")
    elif args.model == 'bloomz':
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz", torch_dtype="auto", device_map="auto")
    elif args.model== 'hf':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)

    logger.info("模型:{} 模型名:{}".format(args.model,args.model_name))

    task_dict={
        #"zbench-basic": os.path.join(os.path.dirname(os.path.abspath(__file__)),"./datasets/basic.json"),
        #"zbench-advanced": os.path.join(os.path.dirname(os.path.abspath(__file__)),"./datasets/advanced.json"),
        #"zbench-domain": os.path.join(os.path.dirname(os.path.abspath(__file__)),"./datasets/domain.json"),
        "regression": os.path.join(os.path.dirname(os.path.abspath(__file__)),"./datasets/regression.json"),
    }

    results=[]

    for task,data_path in task_dict.items():
        avg_score = 0
        with open(data_path,'r',encoding='utf-8') as f:
            dataset = json.load(f)

            for i,data in tqdm(enumerate(dataset)):
                # question response
                question = data['Question'] if 'zbench' in task else data['question']
                question,response = get_result(args.model,question,model,args.model_path, tokenizer)
                # gpt-3.5
                if 'zbench' in task:
                    reference_answer = data["Answers"]["GPT-4"][0][0]
                else:
                    reference_answer = data["reference"]

                if args.evaluate:
                    score = evaluate(question,reference_answer,response)
                else:
                    score = 0.0
                avg_score+= score
                entry= data
                entry['question'] = question
                entry['reply'] =response
                entry['score']=score
                entry['source']=task
                entry['reference']=reference_answer
                results.append(entry)
                #break
            
            avg_score /= len(dataset)
            logger.info("{} 平均得分 {}".format(task,avg_score))
            

    if args.output_format == 'oa_zbench':
        df_to_oa_zbench_json(os.path.join(args.output_path,'result_zb.json'),results)
    elif args.output_format == 'oa_regression':
        #output_path, df, answer_column,reference_answer_column='GPT3.5'
        df_to_oa_regression_json(os.path.join(args.output_path,'result_re.json'),results, args.model_name)
    else:
        raise ValueError('Unkown Format !')
    


def evaluate(question,gpt,answer):
    prompt = "Please give a score to the answer, ranging from 1 to 5. " \
             "If the response does not meet the requirements of the instruction, " \
             "the score is 1. Only return an integer, no additional explanation is needed. " \
             "“Instruction”: {}," \
             "“The response with score 5“:{}," \
             "“Response”:{}"

    extension = 'chatgpt_user'
    #openai.api_key = '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'
    content = prompt.format(question, gpt, answer)
    response = openai_generate_answer([content])
    return response

# 评估模型准确率 参考 "Exploring ChatGPT’s Ability to Rank Content:
# A Preliminary Study on Consistency with Human Preferences"
# def evaluate(logger):
#     prompt = "Please give a score to the answer, ranging from 1 to 5. " \
#              "If the response does not meet the requirements of the instruction, " \
#              "the score is 1. Only return an integer, no additional explanation is needed. " \
#              "“Instruction”: {}," \
#              "“The response with score 5“:{}," \
#              "“Response”:{}"
#
#     extension = 'chatgpt_user'
#     openai.api_key = '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'
#
#     # 平均得分 [1,5]
#     avg_score = 0
#     with open("result.json", 'r', encoding='utf-8') as f:
#         result = json.load(f)
#         for item in tqdm(result):
#             # input of ChatGPT
#             content = prompt.format(item["Question"],item["GPT3.5"][0],item["Answers"])
#             response = generate_answer([{"role": "user", "content": content}])
#             item["Score"] = response
#             avg_score += float(response)
#         # save score in the "result.json"
#         with open("result.json", 'w', encoding='utf-8') as ff:
#             json.dump(result, ff, ensure_ascii=False)
#         avg_score /= len(result)
#         logger.info("处理完毕,共评估{}条记录.当前模型平均得分{}".format(len(result),avg_score))


if __name__ == '__main__':
    args, logger = set()
    main(args,logger)
    # evaluate(logger)