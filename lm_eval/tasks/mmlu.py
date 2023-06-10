from lm_eval.base import MultipleChoiceTask,MultipleChoiceTaskJoint
import datasets
from datasets import load_from_disk
import re
from lm_eval.metrics import mean, subfield_mean



class MMLU(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "data/mmlu"
    DATASET_NAME = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)


    def has_training_docs(self):
        return False
        
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

# class MMLUv2(MultipleChoiceTaskJoint):
#     VERSION = 0
#     DATASET_PATH = "data/mmlu"
#     DATASET_NAME = None
#     KEYSET = ("[A]", "[B]", "[C]", "[D]")

#     def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
#         self.dataset = load_from_disk(self.DATASET_PATH)


#     def has_training_docs(self):
       
#         return False
#     def has_validation_docs(self):
#         return True

#     def has_test_docs(self):
#         return True

#     def validation_docs(self):
#         return map(self._process_doc, self.dataset["validation"])

#     def test_docs(self):
#         return map(self._process_doc, self.dataset["test"])

#     def doc_to_target(self, doc):
#         return " " + doc["choices"][doc["gold"]]

#     def _process_doc(self, doc):
#         def format_example(doc, keys):
#             """
#             Question: <prompt>
#             Choices:
#             [A]. <choice1>
#             [B]. <choice2>
#             [C]. <choice3>
#             [D]. <choice4>
#             Answer:
#             """
#             prompt = "Question: " + doc["question"] + "\nChoices:\n"
#             prompt += "".join(
#                 [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
#             )
#             prompt += "Answer:"
#             return prompt

#         keys = ["[A]", "[B]", "[C]", "[D]"] #keys
#         return {
#             "query": format_example(doc, keys),
#             "choices": keys,
#             "gold": keys.index(doc["answer"])
#             if isinstance(doc["answer"], str)
#             else doc["answer"],
#         }

#     def fewshot_examples(self, k, rnd):
#         # fewshot_examples is not just sampling from train_docs because dev is
#         # in the same distribution as val/test but auxiliary_train isn't

#         if self._fewshot_docs is None:
#             self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

#         return rnd.sample(list(self._fewshot_docs), k)

#     def doc_to_text(self, doc):
#         return doc["query"]

#     def should_decontaminate(self):
#         return True

#     def doc_to_decontamination_query(self, doc):
#         return doc["query"]
    
    
    
class MMLUv2(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = "data/mmlu"
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)
        print(len(self.dataset['test']))


    def has_training_docs(self):
       
        return False
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            [A]. <choice1>
            [B]. <choice2>
            [C]. <choice3>
            [D]. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["[A]", "[B]", "[C]", "[D]"] #keys
        return {
            "fields": doc["fields"],
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

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
        return {"acc": gold==answer, "invalid":invalid, "acc_field":[gold==answer,invalid, doc["fields"]]}
    
    def aggregation(self):
        return {
            "acc": mean,
            "invalid": mean,
            "acc_field": subfield_mean(self.DATASET_PATH),
        }
    
    
        
class MMLUcn(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_PATH = "data/mmlu_cn"
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
        self.dataset = load_from_disk(self.DATASET_PATH)
        print(len(self.dataset['test']))


    def has_training_docs(self):
       
        return False
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            [A]. <choice1>
            [B]. <choice2>
            [C]. <choice3>
            [D]. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["[A]", "[B]", "[C]", "[D]"] #keys
        return {
            "fields": doc["fields"],
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
    
    def process_results(self, doc, results):
        completion = results[0] if len(results[0])>1 else 'INVALID'
        answer = self.extract_answer(completion)
        gold = doc["choices"][doc["gold"]]
        invalid = 0.0
        if hasattr(self, "KEYSET") and answer not in self.KEYSET:
            print("invalid:",answer,completion)
            invalid = 1.0
        return {"acc": gold==answer, "invalid":invalid, "acc_field":[gold==answer,invalid, doc["fields"]]}
    
    def aggregation(self):
        return {
            "acc": mean,
            "invalid": mean,
            "acc_field": subfield_mean(self.DATASET_PATH),
        }
    

# class MMLUcn(MultipleChoiceTaskJoint):
#     VERSION = 0
#     DATASET_PATH = "data/mmlu_cn"
#     DATASET_NAME = None
#     KEYSET = ("[A]", "[B]", "[C]", "[D]")

#     def download(self, data_dir=None, cache_dir=None, download_mode=None):
    
#         self.dataset = load_from_disk(self.DATASET_PATH)


#     def has_training_docs(self):
       
#         return False
#     def has_validation_docs(self):
#         return True

#     def has_test_docs(self):
#         return True

#     def validation_docs(self):
#         return map(self._process_doc, self.dataset["validation"])

#     def test_docs(self):
#         return map(self._process_doc, self.dataset["test"])

#     def doc_to_target(self, doc):
#         return " " + doc["choices"][doc["gold"]]

#     def _process_doc(self, doc):
#         def format_example(doc, keys):
#             """
#             Question: <prompt>
#             Choices:
#             [A]. <choice1>
#             [B]. <choice2>
#             [C]. <choice3>
#             [D]. <choice4>
#             Answer:
#             """
#             prompt = "Question: " + doc["question"] + "\nChoices:\n"
#             prompt += "".join(
#                 [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
#             )
#             prompt += "Answer:"
#             return prompt

#         keys = ["[A]", "[B]", "[C]", "[D]"] #keys
#         return {
#             "query": format_example(doc, keys),
#             "choices": keys,
#             "gold": keys.index(doc["answer"])
#             if isinstance(doc["answer"], str)
#             else doc["answer"],
#         }

#     def fewshot_examples(self, k, rnd):
#         # fewshot_examples is not just sampling from train_docs because dev is
#         # in the same distribution as val/test but auxiliary_train isn't

#         if self._fewshot_docs is None:
#             self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

#         return rnd.sample(list(self._fewshot_docs), k)

#     def doc_to_text(self, doc):
#         return doc["query"]

#     def should_decontaminate(self):
#         return True

#     def doc_to_decontamination_query(self, doc):
#         return doc["query"]
class CEVAL(MMLUcn):
    VERSION = 0
    DATASET_PATH = "data/ceval_val"
    DATASET_NAME = None
    KEYSET = ("[A]", "[B]", "[C]", "[D]")