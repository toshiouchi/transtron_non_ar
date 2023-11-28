# transtron_non_ar
transformer ベースの non-ar TTS の学習に使った ipynb


Result of TTS learning.

We carried out 600epocs TTS machine learning to make inference of mel spectrogram with JSUT 1.1 train 4700, validation 200, test 100 data. Result is https://drive.google.com/file/d/1IgS5KEe-pdHNnOzOuPr7BN9lb5WzTPbi/view

Feature of Program.

TransformerEncoder and TransformerDecoder have convolutional position wise feed forward network. The network has two convolution layers.

Encoder consists of embedding module, three convolutional layers, positional embbeding and TransformerEncoder.

Decoder consists of prenet which has linear projection, ReLU layer and dropout Layer, positional embbeding, TransformerDecoder and feat_out module which has linear projection.

Between encoder and Decoder there are dulation prediction layer and length regulator layer. Both layers imitate fast speech https://github.com/xcmyz/FastSpeech and fast speech2 https://github.com/ming024/FastSpeech2

After decoder there is postnet layer.

A loss is sum of mean squear erro of inferred mel spectrogram and teacher melspectrogram and mean squeare error of inferred duration and teacher duration.

A treatment of prosody

We treat phoneme and prosody. We look on phoneme + prosody as an new phoneme  when there is prosody symbol. Becasu duration model is used. Duration is obtained for phoneme from duration model, but duration is not obtained for prosody from duration mdel. So, when there is prosody symbol, phoneme + prosody as new phoneme is considered and new phoneme has duration which is duration of original old phneme.

A detail Japanese explanation of TTS machine learning  https://qiita.com/toshiouchi/items/668e88e8bf91e154d779

Data preparation program is new_preprocess.py.

Jyputer Notebook script of learning program is ch142_learning_transtron_dp.ipynb.
