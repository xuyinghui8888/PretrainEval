
import os
import pdb
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig,M2M100Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import StoppingCriteria,StoppingCriteriaList
import numpy as np
import time
import tokenizers
import deepspeed 
import torch.nn.functional as F


class ChatCriteria(StoppingCriteria):

    def __init__(self, ):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] < 2:
            return False
        # tokenizer.batch_encode_plus(["Agent:","Agent :"," Agent:","Agent: "],padding=True,max_length=256)
       
        #return (input_ids[0][-2] ==80190 and input_ids[0][-1] ==29) or (input_ids[0][-2]==123263 and input_ids[0][-1] ==29) \
        #or (input_ids[0][-2] ==80190 and input_ids[0][-1] ==915)

        return (input_ids[0][-2] ==10662 and input_ids[0][-1] ==29) or (input_ids[0][-2]==21585 and input_ids[0][-1] ==29) \
        or (input_ids[0][-2] ==10662 and input_ids[0][-1] ==915)


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence.
    3个好处：
    1.支持batch 但是回退到end需要自己进一步操作
    2.支持直接指定sequence
    3.对比是否停的时候，进行decode，放置一样的字符但是id不同的情况
    """

    def __init__(
        self,
        sequence: str,
        tokenizer,
        batch_size: int,
    ):
        
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer
        print('set up stop criteria', sequence,self.sequence_ids )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[
            :, -self.sequence_id_len :
        ]
        
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
       
        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker
    
    def reset(self):
        self.done_tracker = [False] * len(self.done_tracker)


test_text=[
    'Question: A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?\nAnswer:',
    '如何大规模矩阵的逆?\n',
       'Please explain to me what is helium flash?',
    '延缓衰老的最佳实践如下 \n',
    "Q: Jiaran has 5 cups of milk tea. Fengbin takes one. How many left? Answer is: Let's think step by step. "
]
sample_dict = {

    'top_p':
    {
        'num_beams':1,
        'top_p':0.8 ,
        'top_k':0,
        'do_sample': True,
        'num_return_sequences':1,
        'temperature':1.0,
    },


    'greedy':{
        'do_sample' : False,
        'num_beams':1,
        'temperature':1.0,

    },
    
   
    
    'cs':
    {
        'penalty_alpha':0.6, 
        'top_k':4,
        'temperature':2.0,
    },
   
    'top_k':
    {
        'num_beams':1,
        'top_k':20,
        'do_sample': True,
        'num_return_sequences':1,
        'temperature':2.0,
    },
    'beam_search':
    {
        'num_beams':5,
        'do_sample': False,
        'num_return_sequences':5,
        'temperature':2.0,
    }

}



'''
multi-turn witch cache加速:
1.根据q1 generate a1
2.获取当前的past_key_values：past_kv1
3.given past_kv1,  q2 生成合适的casual mask，直接forward， 获得past_kv2
4.given past_kv2, +1个token 开始生成 

'''

'''
past_key_values (`Tuple[Tuple[Tuple[torch.Tensor]]]` returned when `use_cache=True` 
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            tuples (two elements, key tensor and value tensor). 

            The second Tuple is of length `config.n_layers`, 
            third tuple is having 2 tensors(k and v) of shape
            k : [batch_size, num_heads, head_dim, seq_length]
            v: [batch_size, num_heads, seq_length, head_dim]

            举个例子：
            如果一开始的input length是8， batch size =1 ,那么
            第一个k :  32, 128，8
            第二个k :  32, 128, 9

            使用中可以选择需呀的达到自由调整滑窗的结果


            2. if config.is_encoder_decoder=True 2 additional tensors of shape (batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)

            use_cache is re-used because they don't want to new parameter for this, see
            https://github.com/huggingface/transformers/issues/17016

            
'''

# test cache version : start from context should ouput same outputs with 
def debug_multi_turn_with_cache(model,tokenizer):
    # q = "习近平与杨幂孰美?我觉得是杨幂。 .。" 下段的输出应该和单独输入这个一致。
    q1 = "习近平与杨幂孰美?"
    key = 'greedy'
    batch_size = 1

    if key=='cs' :
        raise ValueError('cs can not be used with a cached version')

    v= sample_dict[key]
    generation_config = GenerationConfig(
                early_stopping=True,
                pad_token=model.config.pad_token_id,
                max_new_tokens = 300,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0, 
                #output_scores = True,
                min_new_tokens = 30,
                **v
            )

    pt_batch = tokenizer(
                    [q1],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to('cuda')
    print('pt1', pt_batch.input_ids)
    # check gpu usage with past key value
    outputs = model.generate(**pt_batch,  max_new_tokens=128, generation_config=generation_config,return_dict_in_generate=True)

    print(outputs.sequences[0].size())

    orig_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print(orig_outputs)
    
    past_key_values = outputs.past_key_values

    print(len(past_key_values),past_key_values[0][0][0].size() )
    
    # check the past key value machenism 
    pt_batch2 = tokenizer(
                    ['我觉得是杨幂。'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to('cuda')

    print('pt2', pt_batch2.input_ids)
   
    #past_attention_mask = torch.ones((batch_size, past_key_values[0][0][0].size()[-1]), device=pt_batch2.input_ids.device)
    pt_batch2['attention_mask'] = None
    #torch.cat( past_attention_mask ,pt_batch.attention_mask)
   
    model_inter_outputs = model(
                **pt_batch2,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                past_key_values = past_key_values[0]
            )
    
    
    past_key_values = model_inter_outputs.past_key_values
    print('inter past_key_values: ', past_key_values[0][0].size() )
    print('begin to generate based on past key values:')
    # if batch size==1, unsqueeze to make it [b, n_heads, head_dim, seq_len]
    #generte_past_key_values = past_key_values[0] if len([q1])!=1 else  past_key_values[0].unsqeeze(0) 


    #torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    pt_batch3 = tokenizer(
                    [' .'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to('cuda')

    # get inital attn mask with all previous kv 
    past_attention_mask = torch.ones((batch_size, past_key_values[0][0].size()[-1]+1), device=pt_batch3.input_ids.device)
    pt_batch3['attention_mask'] = past_attention_mask

    outputs = model.generate(**pt_batch3, 
                generation_config=generation_config,
                return_dict_in_generate=True, 
                past_key_values =  past_key_values, max_new_tokens=50)
    
    print(outputs.keys())

    print( len(outputs.past_key_values) )
    print( len(outputs.past_key_values[0]) )
    
    print( outputs.past_key_values[0][0][0].size() )
    print( outputs.past_key_values[0][0][1].size() )

    print( outputs.past_key_values[1][0][0].size() )
    print( outputs.past_key_values[1][0][1].size() )
    

    # deepspeed
    outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print("Output:\n" + 100 * '-')
    import time
    start_time = time.time()
    print(outputs,flush=True)
    print(time.time()-start_time)


# 方案1. 同时修改了框架代码，预先安装的transformer是不行的，如果遇到速度瓶颈走方案一，找我。
def multi_turn_with_cache(model,tokenizer):
    qs = ['The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Gopher, and a human user, called User. In the following interactions, User and Gopher will converse in natural language, and Gopher will do its best to answer User’s questions. Gopher was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins. \n \
    User: Who is prettier, President Xi or Aragaki Yui? \n Gopher',
    'User: I actually think Yang Mi is prettier. \n Gopher',
    'User: on a second thought, President Xi is not bad at all. Tell me which part of him you like the most. \n Gopher',
    'User: please tell me more! \n Gopher',
    'User: I disagree! His hair is black. \n Gopher'] 

    qs = ['The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Agent, and a human user, called User. In the following interactions, User and Agent will converse in natural language, and Gopher will do its best to answer User’s questions. Agent was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins. \n \
    User: Do you know Bohemian Rhapsody? \n Agent',
    'User: Tell me what are the lyrics of that song. \n Agent',
    'User: Can you sing it for me?" \n Agent',
    'User: So tell me. Who is the singer of that song? ".   \n Agent'
    ] 

    qs = [' A human user, called User query a disease. An agent called Agent extracts the disease name and then answer in English: \n \
        Here is an example: \n \
    User: 地中海贫血症的靶点是什么？ \n \
    Agent: Thalassemia \n \
    User: 苯丙酮尿症怎么治疗, 请你给我列出几种常见的治疗方案，并且附上引文参考？ \n \
    Agent: Phenylketonuria or PKU \n \
    User: 请你给我列出几个与阿兹海默症相关的靶点，以及他们的参考文献？\n \
    Agent'
    ] 

    key = 'top_p'
    batch_size = 1

    if key=='cs' :
        raise ValueError('cs can not be used with a cached version')

    v= sample_dict[key]
    generation_config = GenerationConfig(
                early_stopping=True,
                pad_token=model.config.pad_token_id,
                max_new_tokens = 50,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0, 
                #output_scores = True,
                min_new_tokens = 20,
                **v
            )

    past_key_values = None

    for q in qs:
       
        start_time = time.time()
        pt_batch = tokenizer(
                        [q],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to('cuda')

        pt_batch['attention_mask'] = None
        print(pt_batch)
        # check gpu usage with past key value
        model_outputs = model(
                **pt_batch,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                past_key_values = past_key_values
            )

        past_key_values = model_outputs.past_key_values
        
        # 注意！ generate from 有past_key_values 只能take一个token，为了和这个逻辑统一。我们只能给一个token
        pt_batch2 = tokenizer(
                    [':'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to('cuda')
        
        print(pt_batch2)
        
        past_attention_mask = torch.ones((batch_size, past_key_values[0][0].size(-1)+ pt_batch2.input_ids.size(-1) ), 
        device=pt_batch2.input_ids.device)
        pt_batch2['attention_mask'] = past_attention_mask
        
        chat_criteria = ChatCriteria()
        criteria_list = StoppingCriteriaList()
        stop_sequences = ['User:','User :','User: ']
        criteria_list.extend([MultiTokenEOSCriteria(sequence,tokenizer,batch_size) for sequence in stop_sequences ])
        
        outputs = model.generate(**pt_batch2, 
                generation_config=generation_config,
                return_dict_in_generate=True,
                past_key_values =  past_key_values,
                stopping_criteria = criteria_list
                )
        

        sequences_offset = 0
        # 3 termination cases(we assume min_new_token is bigger than 2): 
        # end token
        if outputs.sequences[0][-1]== 2:
            sequences_offset = 1
            past_key_values = outputs.past_key_values[-2]
        # output tokens such as agent:
        elif chat_criteria(outputs.sequences, None) :
            past_key_values = outputs.past_key_values[-3]
            sequences_offset = 2
        # exceed max_length
        else:
            past_key_values = outputs.past_key_values[-1]
            sequences_offset = 0

        sequence = outputs.sequences[0]
        sequence = sequence[:sequence.shape[-1]-sequences_offset]
        outputs = tokenizer.batch_decode([sequence], skip_special_tokens=False)
      
        print("Output:\n" + 100 * '-')
        print(q,flush=True)
        print(outputs,flush=True)
        print(time.time()-start_time)


# 方案二，直接输入之前的qaq pair。
def multi_turn(model,tokenizer):
  
    qs = ['The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Agent, and a human user, called User. In the following interactions, User and Agent will converse in natural language, and Gopher will do its best to answer User’s questions. Agent was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins. \n\
    User: 列出中国最好的三所大学? \nAgent:',
    'User: 嗯，说出三个来。\nAgent:',
    'User: 没有复旦吗？\nAgent:',
    'User: 那你说说复旦的优缺点。\nAgent:'
    ] 


    key = 'top_p'
    batch_size = 1


    v= sample_dict[key]
    generation_config = GenerationConfig(
                early_stopping=True,
                pad_token=model.config.pad_token_id,
                max_new_tokens = 100,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0, 
                #output_scores = True,
                min_new_tokens = 20,
                **v
            )

    past_key_values = None

    history = ''
    
    criteria_list = StoppingCriteriaList()

    stop_sequences = ['User:','User :','User: ']
    criteria_list.extend([MultiTokenEOSCriteria(sequence,tokenizer,batch_size) for sequence in stop_sequences ])
    #criteria_list.append(ChatCriteria())
    for q in qs:
        history += q 
       
        start_time = time.time()
        pt_batch = tokenizer(
                        [history],
                        padding=True,
                        truncation=True,
                        max_length=8192,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to('cuda')


        outputs = model.generate(**pt_batch, 
                generation_config=generation_config,
                return_dict_in_generate=True,
                stopping_criteria = criteria_list
                )

        sequences_offset = 0

        stop_by_cc = False
        for cc in criteria_list:
            if False not in cc.done_tracker:
                sequences_offset = cc.sequence_id_len
                stop_by_cc = True
            cc.reset() # clean the cc as we start next round
        # 3 termination cases(we assume min_new_token is bigger than 2): 
        # end token
        if outputs.sequences[0][-1]== 2:
            sequences_offset = 1
        # output tokens such as agent:
        elif stop_by_cc :
            pass # as sequences_offset is already set
        else:
            # probably exceed max
            sequences_offset = 0
        
        sequence = outputs.sequences[0]
       
        sequence = sequence[pt_batch.input_ids.shape[-1]:sequence.shape[-1]-sequences_offset]
        outputs = tokenizer.batch_decode([sequence], skip_special_tokens=False)

        
        history+= outputs[0] + '\n' #\n more like movie scripts

        print("Output:\n" + 100 * '-')
        print(q,flush=True)
        print(outputs,flush=True)
        print(time.time()-start_time)


def decoding_methods(model,tokenizer, output_scores):

     for k,v in sample_dict.items():
       
        print(f'\n decoding method :{k}\n' + 100 * '-')
        generation_config = GenerationConfig(
                early_stopping=True,
                pad_token=model.config.pad_token_id,
                max_new_tokens = 500,
                eos_token_id=model.config.eos_token_id,
                length_penalty=2.0, 
                #output_scores = True,
                min_new_tokens = 10,
                **v
            )
            
        print(generation_config)
        for question in test_text:

            pt_batch = tokenizer(
                    [question],
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                ).to('cuda')

            #print(pt_batch)
            
            print("Output:\n" + 100 * '-')
            import time
            start_time = time.time()
            print(pt_batch.input_ids.size())
            outputs = model.generate(**pt_batch,  generation_config=generation_config,return_dict_in_generate=True, 
            output_scores=output_scores,output_hidden_states = True ,max_new_tokens=500, min_new_tokens=20)
            
            print('outputs hidden_states len',len(outputs.hidden_states))
            print('outputs hidden_states layers',len(outputs.hidden_states[0]))
            print('outputs hidden_states tensor size',outputs.hidden_states[0][0].size())
            if output_scores:
                beam_indices = None
                if 'beam_indices' in outputs:
                    beam_indices = outputs.beam_indices
                
                # for generated tokens,  score tuple (gen_len, [batch_size, vocab_size ]). 
                # As decoding will postprocess some logtis, here for generation we output the postprocessed probability
                print('output scores size',len(outputs.scores), outputs.scores[1].size(),torch.topk(outputs.scores[0],k=3))
                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores,beam_indices, normalize_logits=True
                )

                input_length = pt_batch.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    # | token | token string | logits | probability
                    print(f"| {tok:6d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")

                # for prompt+generated, manually calculate per-token logp
                output_sentences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

                output_pt_batch = tokenizer(
                    output_sentences,
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                ).to('cuda')


                """
                inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1], as attention is None, casual mask is used
                gpt     \               \
                logits   1 2 3|4 5 6 7 8 9   <- the first token is omitted (some model doesnot have bos)
                cont_toks      4 5 6 7 8 9      
                """

                model_outputs = model(input_ids= output_pt_batch.input_ids[:,:-1])

                log_softmaxes = F.log_softmax(model_outputs.logits, dim=-1)
                
                target_logits = torch.gather(
                    log_softmaxes, 2, output_pt_batch.input_ids[:,1:].unsqueeze(-1)
                ).squeeze(-1)
                print(f'output sequences ids {output_pt_batch.input_ids} \noutput sequences per-token logp {target_logits}')

            outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
            
            print(outputs,flush=True)
            print(time.time()-start_time)
            
            import deepspeed
    
            deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)
            exit()

def main():
    torch.manual_seed(0)  
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    output_scores = True
    output_past_key_values = False
    
    model_checkpoint = "./bloomz-7b1-mt"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                              return_tensors=True,
                                              use_fast=True)
    # tokenizer.padding_side = 'left'  # 
    print(f"Tokenizer: {tokenizer}")

    '''
    from tokenizers.processors import TemplateProcessing
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    '''

    encoding = tokenizer.encode("习近平与杨幂孰美 ? ",padding=True,max_length=128)
    input_tokens  = tokenizer.batch_encode_plus(["User:","User :"," Agent:","Agent: ",'4',' 4'],padding=True,max_length=256)

    print('testing encoding: input_tokens', input_tokens)
    print(encoding)
    print('testing encoding: decode',tokenizer.decode(encoding,skip_special_tokens=False))
   

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,device_map="auto",torch_dtype=torch.float16)
    model.eval()

    #debug_multi_turn_with_cache(model,tokenizer)
    
    #multi_turn(model,tokenizer)
    
    decoding_methods(model,tokenizer,output_scores=True)

   


if __name__ == "__main__":
    main()