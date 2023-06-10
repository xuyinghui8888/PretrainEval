from . medical import MedicalTask
from . medical_context import MedicalTaskContext
import os
DIR = "data/medical_eval"
class USMLE_Step1(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"usmle_step1")

class USMLE_Step2(MedicalTask):
    VERSION = 0
    DATASET_PATH = os.path.join(DIR,"usmle_step2")

class USMLE_Step3(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"usmle_step3")

class USMLE_Step1_Text(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"usmle_step1_text")

class USMLE_Step2_Text(MedicalTask):
    VERSION = 0
    DATASET_PATH = os.path.join(DIR,"usmle_step2_text")

class USMLE_Step3_Text(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"usmle_step3_text")

####### All mmlu task here use the same description as other medical tasks, keep consistency with GPT-4 ##
class MMLU_Clinical_Knowledge(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_clinical_knowledge")

class MMLU_Medical_Genetics(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_medical_genetics")

class MMLU_Anatomy(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_anatomy")

class MMLU_Professional_Medicine(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_professional_medicine")

class MMLU_College_Biology(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_college_biology")

class MMLU_College_Medicine(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"mmlu_college_medicine")
####### All mmlu task here use the same description as other medical tasks, keep consistency with GPT-4 ##

class MedMCQA(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medmcqa_dev_4")

class MedQA_Mainland_5(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medqa_mainland_5")

class MedQA_Mainland_4(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medqa_mainland_4")

class MedQA_Taiwan_4(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medqa_taiwan_4")

class MedQA_US_5(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medqa_us_5")

class MedQA_US_4(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"medqa_us_4")

class Pubmed_QA(MedicalTask):
    DATASET_PATH = os.path.join(DIR,"pubmedqa")

class Pubmed_QA(MedicalTaskContext):
    DATASET_PATH = os.path.join(DIR,"pubmedqa_rr")