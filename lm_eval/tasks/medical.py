from lm_eval.base import MultipleChoiceTask,MultipleChoiceTaskJoint
import datasets
from datasets import load_from_disk
class MedicalTask(MultipleChoiceTaskJoint):
    VERSION = 0
    DATASET_NAME = None
    KEYSET = ("A", "B", "C", "D","E","F", "G", "H", "I","J")
    description = "The following are multiple choice questions (with answers) about medical knowledge.\\n{labeled_examples}{example[:-12]}"


    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_from_disk(self.DATASET_PATH)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

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
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Assistant:
            """
            prompt = "**Question**: " + doc["question"] + "\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "**Answer:**"
            return prompt

        keys = ["A", "B", "C", "D","E","F", "G", "H", "I","J"]  # keys
        choice = doc["choices"]
        keys = keys[:len(choice)]

        return {
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

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = self.description if self.description else  "{labeled_examples} + {example}"

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        #return description + labeled_examples + example
        # print(description)
        return eval("f'"+ description +"'")

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def process_results(self, doc, results):

        completion = results[0] if len(results[0]) > 0 else 'INVALID!'
        answer = self.extract_answer(completion)
        gold = doc["choices"][doc["gold"]]
        # print("result====>",results[0],gold,answer)

        invalid = 0.0
        # if the anser is not in keyset, means it is invalid.
        if hasattr(self, "KEYSET") and answer not in self.KEYSET:
            print("invalid:", answer, completion)
            invalid = 1.0
        return {"acc": gold == answer, "invalid": invalid}

    def extract_answer(self, completion):
        # by default, A] extract to A then format with [A]
        answer = completion[-1]
        keyset = self.KEYSET if hasattr(self, "KEYSET") else set(['A', 'B', 'C', 'D'])
        if answer.strip() not in keyset:
            import re
            match = re.search(r'([A|B|C|D|E|F|G|H|I|J])([\n .\]\)])+', completion)
            if match is None:
                pass
            else:
                # match A] , [0] to get A
                answer = match.group()[0]

        return answer