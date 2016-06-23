#!/usr/bin/env bash

function check_exit_code {
    local status=$?
    if [ $status -ne 0 ]; then
	exit $status
    fi
}


for predictor in inception-v3 alex-owt vgg resnet-50
do
    python train_image.py -p ${predictor} -g -1 -b 1 -i 1
    check_exit_code
done

for predictor in deepspeech2 fc5
do
    python train_audio.py -p ${predictor} -g -1 -b 1 -i 1
    check_exit_code
done

for predictor in c3d
do
    python train_movie.py -p ${predictor} -g -1 -b 1 -i 1
    check_exit_code
done

for predictor in small-lstm big-lstm
do
    python train_text.py -p ${predictor} -g -1 -b 1 -i 1
    check_exit_code
done
