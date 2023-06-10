


import json
import argparse

MEDICAL_TASK_MAP= {
        "medmcqa_dev_4": 0,
        "medqa_ml5": 0,
        "medqa_taiwan_4": 0,
        "medqa_us_4": 0,
        "medqa_us_5": 0,
        "mmlu_anatomy": 0,
        "mmlu_clinical_knowledge": 0,
        "mmlu_college_biology": 0,
        "mmlu_college_medicine": 0,
        "mmlu_medical_genetics": 0,
        "mmlu_professional_medicine": 0,
        "pubmed_rr": 0,
        "usmle1text": 0,
        "usmle2text": 0,
        "usmle3text": 0,
        "medqa_ml4": 0,
        "usmle1": 0,
        "usmle2": 0,
        "usmle3": 0,
    }

MEDICAL_TASK = {
      "name": "inf_medical",
      "K": -1,
      "version": 0,
      "fields": []
        
}


def merge_versions(version, version2merge):
    for key,value in version2merge.items():
        if key in version:
            raise ValueError('key duplicated. Result.json should come from one run')
        version[key] = value

    
def merge_results(result, result2merge, versions2merge, config, light=False):
    """
    修改成如下格式方便前端展示
    "results": {
        "drop": {
        "f1":0.24
        }
    }
    "results": [
    {
      "name": "drop",
      "fields": [
        { "name": "em", "value": 0.2457005033557047 },
        { "name": "em_stderr", "value": 0.004408740972120532 }
      ]
    }
      ],
    """
    formatted_results=[]
    k = config['num_fewshot']
    
    for task,metrics in result2merge.items():
        task_metrics = {}
        task_metrics['name'] =task
        task_metrics['K'] = k 
        task_metrics['version'] = versions2merge[task]
        task_metrics['fields'] = []
        
        if task in MEDICAL_TASK_MAP:
            MEDICAL_TASK_MAP[task] = metrics
            MEDICAL_TASK['K'] = k
            continue
        for key,value in metrics.items():
            
            if 'acc_field' in key:
                add_subfields(  task_metrics['fields'], value)
                
                continue
            
            if light:
                if 'stderr' in key.lower() or 'invalid' in key.lower():
                    continue
            task_metrics['fields'].append({"name":key, "value":value})
            
        formatted_results.append(task_metrics)
    
    result.extend(formatted_results)

# only report acc otherwise too many
def add_subfields(task_fields, subfileds):
    for subfield,metric in subfileds.items():
        task_fields.append({"name":subfield+'_acc', "value": metric['acc'], "parent":'acc'})

def post_process_med(results):
    total_acc=[]
    if MEDICAL_TASK['K']!=-1: #MEDICAL_TASK
        for key,metrics in MEDICAL_TASK_MAP.items():
            if metrics ==0:
                continue
            MEDICAL_TASK['fields'].append({"name":key+"_acc", "value":metrics['acc'],"parent":'acc'})
            total_acc.append(metrics['acc'])
        MEDICAL_TASK['fields'].append({"name":'acc', "value": sum(total_acc)/len(total_acc)})
    
        results.append(MEDICAL_TASK)    
       
            
    

    
def parse_args():
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--input_path", default='gptx66b/result.json')
    parser.add_argument("--output_path", default='gptx66b/result_f.json')
    parser.add_argument("--model_name", type=str, default='gptx66')
    parser.add_argument("--light", action='store_true')
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = {}
    results['results']=[]
    results['versions'] = {}
    
    results['labels'] = {
        "model_name":args.model_name,
        "step":args.step
    }
    with open(args.input_path,'r') as f:
        data = f.read()
        data = data.split('\n\n')
        for idx,one_json in enumerate(data):
            
            if len(one_json)==0:
                continue
            one_result = json.loads(one_json)
            if idx==0:
                results['config'] = one_result['config']

            merge_versions(results['versions'],one_result['versions'])
            merge_results(results['results'],one_result['results'],one_result['versions'],one_result['config'],args.light)
    
    # update K&version to task
    #results.pop('versions')
    
    post_process_med(results['results'])
    print(results)
            
    with open(args.output_path, "w") as f:
        dumped = json.dumps(results, indent=2)
        f.write(dumped)
        