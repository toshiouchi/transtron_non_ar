# transtron_non_ar

Result of TTS learning.

We carried out 600epocs TTS machine learning to make inference of mel spectrogram with JSUT 1.1 train 4700, validation 200, test 100 data. Result is https://drive.google.com/file/d/1IgS5KEe-pdHNnOzOuPr7BN9lb5WzTPbi/view with HiFiGAN https://github.com/jik876/hifi-gan.

Feature of Program.

TransformerEncoder and TransformerDecoder have convolutional position wise feed forward network. The network has two convolution layers. TransformerEncoder works as self attention.

Encoder consists of embedding module, three convolutional layers, positional embbeding and TransformerEncoder.

Between encoder and Decoder there are duration prediction layer and length regulator layer. Both layers imitate fast speech https://github.com/xcmyz/FastSpeech and fast speech2 https://github.com/ming024/FastSpeech2

Decoder consists of prenet module, positional embbeding, TransformerDecoder and feat_out module which has linear projection. The prenet module consists of two layers of linear projection, ReLU and dropout. Inputs of decoder are two values, one is encoder output as soruce input of TransformerDecoder and another is length regulator output as target input of TransformerDecoder. TransformerDecoder works as cross attention. 

After decoder there is postnet layer which has 5 convolution layers.

A loss is sum of mean squear error of inferred mel spectrogram and teacher melspectrogram and mean squeare error of inferred duration and teacher duration.

A treatment of prosody

We treat phoneme and prosody. We look on phoneme + prosody as an new phoneme  when there is prosody symbol. Becuase duration model do not treat prosody. There are durations for phoneme from the model, but there are not durations for prosody from the model. So, when there is prosody symbol, phoneme + prosody is considered as new phoneme and new phoneme has duration which is one of original old phoneme.

A detail Japanese explanation of TTS machine learning  https://qiita.com/toshiouchi/items/668e88e8bf91e154d779

Data preparation program is new_preprocess.py.

Jupyter Notebook script of learning program is ch142_learning_transtron_dp.ipynb.

