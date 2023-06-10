import os
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time
import requests
from lm_eval.api import inf_generate_until,inf_likelihood
from typing import List, Mapping, NewType, Optional, Tuple, Union

def generate_answer(messages):

    url = 'http://10.193.5.100:5678/test/v1/api/evaluate'
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
    # 这个是认真的吗。。。。
    answer = float(rjson['data']['questionResults'][0]['modelResults'][0]['answers'][0])
    return answer
      


def get_continuation_logprobs(response, ctxlen):
   
    logprobs = response[0]
    offsets = response[1]
    
    continuation_idx = None
    for idx, offset in enumerate(offsets):
        if offset>= ctxlen:
            continuation_idx = idx
            break
    
    if continuation_idx is None:
        raise ValueError('continuation_idx is None, Check your input!')
    # text 1 2 3 4 5
    # token 1 2  4 5
    # offset 0,1,3,4
    # logp -0.5 -0.6 0.1
    # cxt_len=3, then should sum 4,5
    
    continuation_logprobs = sum(logprobs[continuation_idx-1:])

    return continuation_logprobs


class INFLM(BaseLM):
    REQ_CHUNK_SIZE = 8

    def __init__(self, url, max_length=2048,truncate=False, batch_size=1, max_gen_toks=30):
        """
        directly use inf api as model.
        support generate_until & likelyhood
        
        """
        super().__init__()

        self.max_len = max_length
        self.engine = url
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

        self.vocab_size = self.tokenizer.vocab_size

        # to make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
       
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(
            ["<|endoftext|>"]
        )[0]
        self._batch_size = batch_size
        self.url = url
        self._max_gen_toks = max_gen_toks

       
    @property
    def eot_token_id(self):
        # this is only for rolling likelihood. should not be used against infapi
        raise ValueError('rolling likelihood should not be used against infapi')
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        # change this accordingly based-on differnet model settings.
        return self.max_len

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()
        
    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    def tok_encode(self, string: str):
        raise ValueError('should not be called!')
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        raise ValueError('should not be called!')
        return self.tokenizer.decode(tokens)

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for context, continuation in requests:
            
            # Fill empty contexts with the EOT token.
            context = f"{self.eot_token}" if len(context) == 0 else context 
            
            
            context_enc = context
            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation_enc = continuation
            #.lstrip()
            
            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
            
        return self._loglikelihood_tokens(new_requests)
    
    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        
        # (context, continuation), context_enc, continuation_enc

        for chunk in tqdm(
            list(utils.chunks(requests, self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                
                inp = (context_enc+continuation_enc)[-(self.max_length + 1):] # again for infapi max len is text len
            
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )
                
                
              
                inps.append(inp)
                ctxlens.append(ctxlen)
            
           
            response = inf_likelihood(
                messages=inps,
                max_new_token=0,
                output_scores=True,
                url = self.url,
                temperature=1.0,
                topp=1,
                disable_prompt=True
            )
            

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                response.items(), ctxlens, chunk
            ):
                resp = resp[1][0] # (logprobs, offsets)
                answer = get_continuation_logprobs(resp, ctxlen)
                # cannot tell if greedy, just return False
                res.append((answer,False))
                
                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
           
        return res

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = x[0]
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size))
        ):
            inps = []
            for context, _ in chunk:
                context_enc = context
                # 这里因为问infapi是直接发text，所以max_length是string的length
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)
            
            
            response = inf_generate_until(
                messages=inps,
                max_new_token=self.max_gen_toks,
                #temperature=0.0,
                output_scores=False,
                until=until,
                url = self.url,
                temperature=0.35,
                topp=0.01,
                disable_prompt=False
            )
            #print('===>',inps,response)
            
            for resp, (context, until_) in zip(response.items(), chunk):
                #这个是格式 batch_id+答案 (0, ['世界上最大的动物是蓝鲸。\n', '世界上最大的动物是蓝鲸。蓝鲸是一种'])
                print(resp)
                s = resp[1][0]

                for term in until_:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until_), s)

                res.append(s)
        
        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
