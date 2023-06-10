


# env
conda create -n har python=3.8 pip
conda activate har
pip install -e . 

# 有时候会根据命令 install 1.12.1 but default install will not work for A100
# for A100 pip uninstall torch then pip3 install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# 有些镜像不好用，可以这样 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install sacrebleu
pip install omegaconf
pip install pycountry
pip install accelerate
pip install pytablewriter

# 如果还嫌推理慢就加这个。然后联系我，还能快个20%
pip install deepspeed

# 如果不全自己在试试 手动->微笑+抱歉

# 在pai上
# git 
sudo apt-get install netcat
# 参考yichen的文档配置gitlab

cd transformers/
pip install e .