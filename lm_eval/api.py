import requests
from collections import defaultdict


def openai_generate_answer(messages):

    url = 'http://10.193.7.200:8080/openassistant/ask'
    url = 'http://172.16.3.9:8081/openassistant/ask'
    url = 'http://10.12.135.96:8081/openassistant/ask'
    headers = {'Content-Type': 'application/json'}

    # get user
    # [USERNAME], i.e.: myuser
    # [PASSWORD], i.e.: itspassword

    data = {
        "questions": messages,
        "models": [
            {
                
                "params": {
                    "model":"gpt-3.5-turbo"
                },
                "name": "gpt",
                "api": True,
            }
        ]
    }

    r = requests.post(url, json=data, headers=headers)
    rjson= r.json()
    print(rjson)
    # 这个是认真的吗。。。。
    answer = float(rjson['data']['questionResults'][0]['modelResults'][0]['answers'][0])
    return answer
      

def inf_generate_until(messages, output_num=1, until=['<\s>'], max_new_token=8, output_scores=False, url=None, 
                       topp=0.85, temperature=1.0, disable_prompt=False):

    url = url if url is not None else 'http://10.193.5.100:5678/test/v1/api/evaluate' 
    headers = {'Content-Type': 'application/json'}

    # get user
    # [USERNAME], i.e.: myuser
    # [PASSWORD], i.e.: itspassword

    stop_words = ["Human:"]
    stop_words.extend(until)
    batch_size = len(messages)
    data = {
        "input_texts":messages,
        "topp":topp, 
        "temperature": temperature,
        "output_scores":output_scores, 
        "output_num":output_num, 
        "max_new_token":max_new_token,
        "stop_words":stop_words,
        "disable_prompt": disable_prompt
    }
    
    
    r = requests.post(url, json=data, headers=headers)
    rjson= r.json()
    print('response',messages,rjson,flush=True)
    answers=defaultdict(list)
    for i in range(output_num):
        for b in range(batch_size):
            answers[b].append(rjson['data'][b][i]['output'])
    
    #answer = rjson['data'][0]['output'].split(']')[0]
    return answers
 
def inf_likelihood(messages, output_num=1, until=['<\s>'], max_new_token=0, output_scores=True, url=None, 
                       topp=0.85, temperature=1.0, disable_prompt=True):

    url = url if url is not None else 'http://10.193.5.100:5678/test/v1/api/evaluate' 
    headers = {'Content-Type': 'application/json'}

    # get user
    # [USERNAME], i.e.: myuser
    # [PASSWORD], i.e.: itspassword
    assert output_num==1, "in likelihood mode, it consumes more memory. Set output_num to 1"
    assert output_scores==True, "in likelihood mode, output likelihood"
    assert disable_prompt==True, "in likelihood mode, disable prompt"
    batch_size = len(messages)
    data = {
        "input_texts":messages,
        "topp":topp, 
        "temperature": temperature,
        "output_scores":output_scores, 
        "output_num":output_num, 
        "max_new_token":max_new_token,
        "stop_words": ["Human:"].extend(until),
        "disable_prompt": disable_prompt
   }

    r = requests.post(url, json=data, headers=headers)
    rjson= r.json()
    answers=defaultdict(list)
    """
      {'count': 1, 
  'data': [[{'input_log_probs': [-41.2640266418457, -6.809728622436523, -4.120052337646484, -2.4455461502075195, -2.856555461883545, -2.4689080715179443], 'input_offsets': [0, 1, 3, 4, 5, 6, 7], 'input_token_num': 7, 'input_tokens': ['1', '+1', '是', '几', '？', '\n', '2'], 'output': '+1', 'output_log_probs': [-0.5271965861320496], 'output_offsets': [0], 'output_token_num': 1, 'output_tokens': ['+1']}]], 
  'error_code': 0, 
  'error_message': 'success', 
  'latency': '1057.71ms'}
    """
    for i in range(output_num):
        for b in range(batch_size):
            elem = rjson['data'][b][i]
            answers[b].append( (elem['input_log_probs'], elem['input_offsets']) )
    
    return answers


import json

"""
{
        "origin": "z-bench_advanced_v0406",
        "lang": "zh",
        "prompt": {
            "text": "\u5982\u679c\u6211\u5728\u7a7a\u4e2d\u5782\u76f4\u629b\u7403\uff0c\u7403\u6700\u521d\u5411\u54ea\u4e2a\u65b9\u5411\u884c\u8fdb\uff1f",
            "replies": [
                {
                    "text": "\u5f53\u4f60\u5728\u7a7a\u4e2d\u5782\u76f4\u629b\u7403\u65f6\uff0c\u7403\u6700\u521d\u884c\u8fdb\u7684\u65b9\u5411\u662f\u5411\u4e0a\u7684\uff0c\u4e5f\u5c31\u662f\u548c\u5730\u9762\u5782\u76f4\u7684\u65b9\u5411\u3002\u5728\u7403\u5230\u8fbe\u6700\u9ad8\u70b9\u65f6\uff0c\u5b83\u7684\u8fd0\u52a8\u65b9\u5411\u4f1a\u53d1\u751f\u53d8\u5316\uff0c\u5e76\u5f00\u59cb\u4e0b\u964d\u3002",
                    "model_name": "gpt-3.5"
                },
                {
                    "text": "\u5f53\u4f60\u5728\u7a7a\u4e2d\u5782\u76f4\u5411\u4e0a\u629b\u7403\u65f6\uff0c\u7403\u6700\u521d\u7684\u65b9\u5411\u662f\u5411\u4e0a\u3002\u5b83\u5c06\u6cbf\u7740\u5782\u76f4\u65b9\u5411\u53d7\u5230\u91cd\u529b\u7684\u4f5c\u7528\u800c\u51cf\u901f\uff0c\u76f4\u81f3\u5728\u5176\u6700\u9ad8\u70b9\u901f\u5ea6\u964d\u4e3a0\u3002\u7136\u540e\uff0c\u7403\u5c06\u5f00\u59cb\u6cbf\u76f8\u53cd\u7684\u5782\u76f4\u65b9\u5411\u5411\u4e0b\u52a0\u901f\uff0c\u6700\u7ec8\u56de\u5230\u5730\u9762\u3002\u5728\u6574\u4e2a\u8fc7\u7a0b\u4e2d\uff0c\u7403\u7684\u8fd0\u52a8\u53d7\u5230\u5730\u7403\u91cd\u529b\u7684\u5f71\u54cd\u3002",
                    "model_name": "gpt-4"
                },
                {
                    "text": "\u5982\u679c\u6211\u5728\u7a7a\u4e2d\u5782\u76f4\u629b\u7403\uff0c\u90a3\u4e48\u7403\u6700\u521d\u4f1a\u5411\u4e0b\u65b9\u884c\u8fdb\u3002\u7531\u4e8e\u91cd\u529b\u7684\u4f5c\u7528\uff0c\u7403\u5c06\u4f1a\u843d\u5411\u5730\u9762\uff0c\u5e76\u4e14\u671d\u7740\u5730\u9762\u7684\u65b9\u5411\u79fb\u52a8\u3002\u56e0\u6b64\uff0c\u5982\u679c\u6211\u5728\u7a7a\u4e2d\u5782\u76f4\u629b\u7403\uff0c\u7403\u6700\u521d\u4f1a\u5411\u4e0b\u65b9\u79fb\u52a8\uff0c\u7136\u540e\u518d\u6839\u636e\u5177\u4f53\u60c5\u51b5\u51b3\u5b9a\u5b83\u5c06\u671d\u54ea\u4e2a\u65b9\u5411\u884c\u8fdb\u3002",
                    "model_name": "\u6587\u5fc3\u4e00\u8a00"
                }
            ]
        }
    },
"""

def df_to_oa_zbench_json(output_path, df):
    # df format: ['Question', "Model1", 'GPT3.5', 'Score','Source']

    columns = set(df.columns.values)
    columns.remove('Question')
    columns.remove('Score')
    columns.remove('Source')

    entries = []
    for index, row in df.iterrows():
        one_entry = {
            "origin":row['Source'],
            "lang":"zh",
            "prompt": {
                "text": row['Question'],
                "replies":[
                    {"text": row[c], "model_name":c} for c in columns
                ]
            }
        }

        entries.append(one_entry)

    
    with open(output_path,"w", encoding='utf-8') as jsonfile:
        json.dump(entries,jsonfile,ensure_ascii=False)

"""
[
    {
        "lang": "en/zh",
        "origin": "bench_xxx",
        "correct_answer": "正确答案",
        "prompt": "Q",
        "reply": "A",
        "reply_model_name": "gpt",
        "description": "xx",
        "type": "",
        "meta": {"id": "123"}
    }
]
"""
def df_to_oa_regression_json(output_path, results, answer_model_name):
    # df format: [id, 'Question', "Answer", 'GPT3.5', 'Score','Source']

    entries=[]
    for row in results:
        one_entry = {
            "origin":row['source'],
            "prompt": row['question'],
            "reply":row['reply'],
            "reply_model_name": answer_model_name,
            "correct_answer": row['reference'],
            "lang": row.get('lang',""),
            "description": row.get('description',""),
            "type": row.get('type',""),
            "meta": {
                "id":row.get('id',"")
            }
        }
        print(one_entry)
        entries.append(one_entry)

    
    with open(output_path,"w", encoding='utf-8') as jsonfile:
        json.dump(entries,jsonfile,ensure_ascii=False)

    


