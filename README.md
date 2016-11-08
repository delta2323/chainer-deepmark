# Deepmark-Chainer
Evaluation scripts of Chainer for [deepmark](https://github.com/DeepMark/deepmark)

## Usage

```
python evaluate/train_image.py
python evaluate/train_audio.py
python evaluate/train_movie.py
python evaluate/train_text.py
```

## Architecture Support Status

|Category|Arcitecture|Status|Comment|
|---|---|---|---|
|Image|InceptionV3-batchnorm|Y||
||Alexnet-OWT|Y||
||VGG-D|Y||
||ResNet50|Y||
|Video|C3D|Need fix|Issue #5, waiting for MaxPoolingND (Chainer issue #1353)|
|Audio|DeepSpeech2|Y|Need check if the architecture is same as Torch deepmark's|
||MSR 5FC layer|Teporary implementation||
|Text|BigLSTM|Working|Issue #2, PR #11|
||SmallLSTM|Working|Issue #19|

## LICENSE

MIT (See `LICENSE` file)
