# MAC-NETWORK

Forked from [rosinality/mac-network-pytorch](https://github.com/rosinality/mac-network-pytorch) which is based on  [Memory, Attention and Composition (MAC) Network for CLEVR from Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067).

This fork fixes some of the problems in the rosinality's implementation, such as adding two missing projections, and the addition of gradient clipping and a LR scheduler.
It also cleans up the code and makes it more easily modified by further separating the model into components and using [yacs](https://github.com/rbgirshick/yacs) for all configuration parameters.


### 1. Install requirements

Run the following command to install missing dependencies:

~~~bash
pip install -r requirements.txt
~~~


### 2. Preprocess questions and images

~~~bash
python preprocess.py [CLEVR directory]
python image_feature.py [CLEVR directory]
~~~


### 3. Train the model

~~~bash
python train.py DATALOADER.FEATURES_PATH [CLEVR directory]
~~~

Alternatively, change the default path to the pre-computed features in the default configuration (under config/defaults) to be able to just run `python train.py` for training.

To experiment with the model without changing code just create a configuration file and pass its location as an argumet to `train.py`:


~~~bash
# a sample configuration file is provided at configs/train.yaml 
python train.py --config-file=configs/train.yaml
~~~