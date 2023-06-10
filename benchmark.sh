
MODEL=${1}

echo ${1}
NUM_GPU=8
OUTPUT_PATH=${2}
MAX_LEN=8192
DTYPE=bfloat16

#NCCL_DEBUG=INFO 
# 有一些模型比如Llama， batch_size必须为1，因为目前开源的tokenizer，没有pad id。
mkdir ${OUTPUT_PATH}

torchrun --nproc_per_node=${NUM_GPU} main.py --num_fewshot 0 --model hf-causal --model_args pretrained=${MODEL},dtype=${DTYPE},max_length=${MAX_LEN} \
--tasks drop --batch_size=1 --distributed --output_path=${OUTPUT_PATH}  --save_examples  >>${OUTPUT_PATH}/b.log  \

torchrun --nproc_per_node=${NUM_GPU} main.py \
--num_fewshot 0 \
--model hf-causal \
--model_args pretrained=${MODEL},dtype=${DTYPE},max_length=${MAX_LEN} \
--tasks truthfulqa_mc,hellaswag --batch_size=1 --distributed --output_path=${OUTPUT_PATH}  --save_examples >>${OUTPUT_PATH}/b.log \

torchrun --nproc_per_node=${NUM_GPU} main.py --num_fewshot 0 --model hf-causal --model_args pretrained=${MODEL},dtype=${DTYPE},max_length=${MAX_LEN} \
 --tasks winogrande,gsm8k,arc_challenge,mmluv2,clue_c3,clue_wsc2020,mmlucn \
 --batch_size=1 --distributed --output_path=${OUTPUT_PATH}  --save_examples >>${OUTPUT_PATH}/b.log \



# PILE example
#torchrun --nproc_per_node=8 main.py --num_fewshot 0 --model hf-causal --model_args  \
#pretrained=${MODEL_PATH},dtype=bfloat16,max_length=4096  --tasks  zh_wiki,pile_wikipedia --batch_size=1 --distributed \
#--output_path=${OUTPUT_PATH} >>${OUTPUT_PATH}/b.log
