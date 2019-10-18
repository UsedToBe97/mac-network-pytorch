# MAC-NETWORK

Forked from [rosinality/mac-network-pytorch](https://github.com/rosinality/mac-network-pytorch) which is based on  [Memory, Attention and Composition (MAC) Network for CLEVR from Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067).

This fork is born after observing that existing pytorch implementations were performing weel below what they should according to the paper and the oficial tensorflow implementation. Following the later the changes done to the implementation this repository was forked from are:

1. Fixed dropout values (for each of now multiple dropout sectors).
1. Fixed variational dropout mask generation and aplication for memory and removed it for control.
1. Fixed LSTM dimensions.
1. Fixed bidirectional RNN question vector.
1. Added dropout to convolutional layers.
1. Fixed weight initializations.
1. Added missing controlUnit projection.
1. Added two missing projections to readUnit (and respective activations).
1. Removed unnecesary projection in writeUnit.
1. Doubled layers in classifier_out.
1. Many more things I forgot to write here...


This implementation also cleans up the code and makes it more easily modified by further separating the model into components, using pytorch 1.0's `device`, and using [yacs](https://github.com/rbgirshick/yacs) for all configuration parameters.



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