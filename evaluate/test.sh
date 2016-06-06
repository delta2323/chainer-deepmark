#!/usr/bin/env bash

for predictor in inception-v3 alex vgg resnet-50
do
    python train_image.py -p ${predictor} -g -1 -b 2
done

for predictor in deepspeech2 fc5
do
    python train_audio.py -p ${predictor} -g -1 -b 2
done

for predictor in c3d
do
    python train_movie.py -p ${predictor} -g -1 -b 2
done

for predictor in small_lstm big_lstm
do
    python train_text.py -p ${predictor} -g -1 -b 2
done