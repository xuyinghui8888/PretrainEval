
from lm_eval.base import MultipleChoiceTask,MultipleChoiceTaskJoint
import datasets
from datasets import load_from_disk
from lm_eval.base import Task, rf
from lm_eval.metrics import mean

class CLUE_C3(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = "data/clue/c3"
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)


    def has_training_docs(self):
       
        return True
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Context: <prompt>
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Context: " + '\n'.join(doc["context"]) + "\nQuestion: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choice"])]
            )
            prompt += "Answer:"
            
            return prompt

        keys = ["[A]", "[B]", "[C]", "[D]"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": doc["choice"].index(doc["answer"])
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["train"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]



class CLUE_WSC2020(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = "data/clue/cluewsc2020"
    DATASET_NAME = None
    KEYSET = ("是", "否")

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)


    def has_training_docs(self):
       
        return True
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def extract_answer(self,completion):
        # bloom default会说 “是的” 和 不。。。
        completion = completion.lower().strip()
        answer_dict ={
            "是的": "是",
            "不": "否",
            "yes":"是",
            "no": "否",
            "不是": "否"
        }
        completion = answer_dict[completion] if completion in answer_dict else completion
        return f"{completion}"

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Context: "<prompt>"
            Question: 上述句子中，“span2”是指“span1”吗？ 回答是或者否。
            Answer: 是or否
            """
            
            prompt = (
                f'Context: \"{doc["text"]}\"\n'
            f'Question: 上述句子中，\"{doc["target"]["span2_text"]}\"是指\"{doc["target"]["span1_text"]}\"吗？回答是或否。\n'
            f'Answer:'
            )

            
            
            return prompt

        keys = ["是", "否"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": doc["label"]
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["train"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def get_stopword(word):
        return ['.','\n','。']


"""
DRCD answer is an extraction-based reading comprehension task.
Sometimes the model answer is correct but will not be consider as EM.
"""
class CLUE_DRCD(Task):
    VERSION = 0
    DATASET_PATH = "data/clue/drcd"
    DATASET_NAME = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)

    def construct_requests(self, doc, ctx):
        
        conts = [rf.greedy_until(ctx, [".","\n"])]
        return conts

    def has_training_docs(self):
       
        return True
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False
    
    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "id": doc["id"],
            "passage": doc["context"],
            "question": doc["question"],
            "answers": doc["answers"]['text']
        }


    def process_results(self, doc, results):
        
        preds, golds = results[0], doc["answers"]
        max_em = 0
        for gold_answer in golds:
            exact_match = 1.0 if preds.strip()==gold_answer.strip() else 0.0
            if gold_answer[0].strip():
                max_em = max(max_em, exact_match)
        return {"em": max_em}


    def doc_to_text(self, doc):
        return f"Passage: {doc['passage']}\nQuestion: {doc['question']}\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"] + " " + doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answers"][0]

    def get_stopword(word):
        return ['.','\n','。']


    def aggregation(self):
        return {"em": mean}

    def higher_is_better(self):
        return {"em": True}
