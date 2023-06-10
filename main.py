import argparse
import json
import logging
import fnmatch

from lm_eval import tasks, evaluator
from lm_eval.utils import print_rank_0
logging.getLogger("openai").setLevel(logging.WARNING)

from transformers import AutoTokenizer # AutoModelForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import torch

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--distributed",  action="store_true")
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--save_examples", action="store_true", help="save examples of evaluation")
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()
        
    # register Model for liushan
    # 如果想评测我们出产的与训练模型，需要做如下工作
    # 找到configuration以及modeling的代码，然后注册
    
    '''
    from configuration_bloom import BloomV2Config
    from modeling_bloom import BloomV2ForCausalLM
    from configuration_gptx import GPTxConfig
    from modeling_gptx import GPTxForCausalLM

    
    AutoConfig.register("bloomv2", BloomV2Config)
    AutoModel.register(BloomV2Config, BloomV2ForCausalLM)
    AutoModelForCausalLM.register(BloomV2Config, BloomV2ForCausalLM)
    
    
    AutoConfig.register("GPTx", GPTxConfig)
    AutoModel.register(GPTxConfig, GPTxForCausalLM)
    AutoModelForCausalLM.register(GPTxConfig, GPTxForCausalLM)
    '''

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    import os
    import torch.distributed as dist
    if args.distributed:
        if args.output_path is None:
            raise ValueError("runing with dist mode must specify output path for saving temp calculations.")
        dist.init_process_group(backend="gloo")
    elif "LOCAL_RANK" in os.environ:
        raise ValueError('Local RANK is set but distributed is not set. Something must be wrong!')
    

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        '''
        tasks_set = set(args.tasks.split(","))
        if 'mmlu' in tasks_set:
            tasks_set.remove('mmlu')
            from lm_eval.tasks.hendrycks_test import SUBJECTS
            tasks_set = list(tasks_set)
            tasks_set.extend( [f"hendrycksTest-{sub}" for sub in SUBJECTS])
        tasks_set = list(tasks_set)
        '''
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    task_names = sorted(task_names)
    print(f"Selected Tasks: {task_names}")
    
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        output_path=args.output_path,
        save_examples=args.save_examples
    )
    
    label_dict={}
    
    label_dict['model_name']=args.model_name
    label_dict['step']=args.step
    results['label'] = label_dict
    
   
    dumped = json.dumps(results, indent=2)
    print_rank_0(dumped)

    rank_0 = False
    if args.output_path:
        import os
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                rank_0 = True
        else:
            rank_0 = True
            
    if rank_0:
        
        with open(os.path.join(args.output_path,'result.json'), "a") as f:
            f.write(dumped+'\n\n')

        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
        )
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
