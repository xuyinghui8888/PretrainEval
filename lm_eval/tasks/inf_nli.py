import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import datasets
from datasets import load_from_disk


class Med(Task):
    VERSION = 0
    DATASET_PATH = "data/med"
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None): 
        self.dataset = load_from_disk(self.DATASET_PATH)
        print(len(self.dataset))
    
    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return False
    
    def has_test_docs(self):
        return True
    
    def test_docs(self):
        if self.has_test_docs():
            return self.dataset
        
    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["premise"],
            doc["hypothesis"].strip()
            + ("" if doc["hypothesis"].strip().endswith(".") else "."),
        )
    
    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])
    
    
    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false
    
    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}
    
    
    def higher_is_better(self):
        return {"acc": True}
    
    def aggregation(self):
        return {"acc": mean}