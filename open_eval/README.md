
## quick start

```python
python open_eval.py --model your_model_name
```

--model 指定模型名字 默认是hf

--model_path 指定模型加载路径

--output_path 指定模型结果的输出路径 默认是当前路径下 result.json

这个脚本使用的时候，需要按照prompt等进行改造，比如模型的适配、模型接口的适配等。

## Requirement

pip install protobuf==3.20.0 icetk cpm_kernels

pip install argparse

pip install logging

## 后续

目前看GPT评测的一致率在60～70%之前。估计还是要对接标注平台。

那么后续就以csv的形式走。

