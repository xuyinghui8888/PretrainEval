"""
LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning
https://arxiv.org/pdf/2007.08124.pdf

LogiQA is a dataset for testing human logical reasoning. It consists of 8,678 QA
instances, covering multiple types of deductive reasoning. Results show that state-
of-the-art neural models perform by far worse than human ceiling. The dataset can
also serve as a benchmark for reinvestigating logical AI under the deep learning
NLP setting.

Homepage: https://github.com/lgw863/LogiQA-dataset
"""
import inspect
import lm_eval.datasets.logiqa.logiqa
from lm_eval.base import MultipleChoiceTask, MultipleChoiceTaskJoint
import datasets 
import os 
from datasets import load_from_disk
from lm_eval.metrics import mean

_CITATION = """
@misc{liu2020logiqa,
    title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
    author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
    year={2020},
    eprint={2007.08124},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class LogiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.logiqa.logiqa)
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
            Passage: <passage>
            Question: <question>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Passage: " + doc["context"] + "\n"
            prompt += "Question: " + doc["question"] + "\nChoices:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Answer:"
            return prompt

        choices = ["a", "b", "c", "d"]
        return {
            "passage": doc["context"],  # Used for decontamination
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"]),
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"]
    

class LogiQAV2(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = 'data/logiqajoint'
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None): 
        self.dataset = load_from_disk(self.DATASET_PATH)
        print(len(self.dataset['test']))
        
    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return False
    
    def has_test_docs(self):
        return True
    
    def test_docs(self):
        return map(self._process_doc, self.dataset['test'])
    
    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]
    
    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Query: <prompt>
            Choices:
            [A]. <choice1>
            [B]. <choice2>
            [C]. <choice3>
            [D]. <choice4>
            Answer:
            """
            prompt = "Passage: " + doc["passage"] + "\n"
            prompt += "Question: " + doc["question"] + "\nChoices:\n"
            
            prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])])
            prompt += "Answer:"
            return prompt
        keys = ["[A]", "[B]", "[C]", "[D]"] 
        
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["answer"]) if isinstance(doc["answer"], str) else doc["answer"]
        }
    
    
    def doc_to_text(self, doc):
        return doc["query"]
    
    def should_decontaminate(self):
        return True
    
    def doc_to_decontamination_query(self, doc):
        return doc["query"]
    
    
    def extract_answer(self,completion):
        # by default, A] extract to A then format with [A]
        answer = completion[-1]
        
        if answer.strip() not in set(['A','B','C','D']):
            import re
            match = re.search(r'[A|B|C|D]', completion)
            if match is None:
                pass
            else:
                answer = match.group()
           
        return f"[{answer}]"
    
    def process_results(self, doc, results):
        completion = results[0] if len(results[0])>1 else 'INVALID'
        answer = self.extract_answer(completion)
        gold = doc["choices"][doc["gold"]]
        invalid = 0.0
        if hasattr(self, "KEYSET") and answer not in self.KEYSET:
            print("invalid:",answer,completion)
            invalid = 1.0
        return {"acc": gold==answer, "invalid":invalid}
    
    def aggregation(self):
        return {
            "acc": mean,
            "invalid": mean
        }
    
class LogiQA_ZH(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = 'data/logiqa_zhjoint'
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None): 
        self.dataset = load_from_disk(self.DATASET_PATH)
        print(len(self.dataset['test']))
        
    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return False
    
    def has_test_docs(self):
        return True
    
    def test_docs(self):
        return map(self._process_doc, self.dataset['test'])
    
    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]
    
    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Query: <prompt>
            Choices:
            [A]. <choice1>
            [B]. <choice2>
            [C]. <choice3>
            [D]. <choice4>
            Answer:
            """
            prompt = "Passage: " + doc["passage"] + "\n"
            prompt += "Question: " + doc["question"] + "\nChoices:\n"
            
            prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])])
            prompt += "Answer:"
            return prompt
        keys = ["[A]", "[B]", "[C]", "[D]"] 
        
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["answer"]) if isinstance(doc["answer"], str) else doc["answer"]
        }
    
    
    def doc_to_text(self, doc):
        return doc["query"]
    
    def should_decontaminate(self):
        return True
    
    def doc_to_decontamination_query(self, doc):
        return doc["query"]
    
    
    def extract_answer(self,completion):
        # by default, A] extract to A then format with [A]
        answer = completion[-1]
        
        if answer.strip() not in set(['A','B','C','D']):
            import re
            match = re.search(r'[A|B|C|D]', completion)
            if match is None:
                pass
            else:
                answer = match.group()
           
        return f"[{answer}]"
    
    def process_results(self, doc, results):
        completion = results[0] if len(results[0])>1 else 'INVALID'
        answer = self.extract_answer(completion)
        gold = doc["choices"][doc["gold"]]
        invalid = 0.0
        if hasattr(self, "KEYSET") and answer not in self.KEYSET:
            print("invalid:",answer,completion)
            invalid = 1.0
        return {"acc": gold==answer, "invalid":invalid}
    
    def aggregation(self):
        return {
            "acc": mean,
            "invalid": mean
        }
    