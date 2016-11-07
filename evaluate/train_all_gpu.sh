#!/usr/bin/env bash

for cache in none memory disk
do
    for predictor in inception-v3 alex-owt vgg-d resnet-50
    do
	python train_image.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache}
	python train_image.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache} -c
    done

    for predictor in deepspeech2 fc5
    do
	python train_audio.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache}
	python train_audio.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache} -c
    done

    for predictor in c3d
    do
	python train_movie.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache}
	python train_movie.py -p ${predictor} -g 0 -b 2 -i 2 -C ${cache} -c
    done

    for predictor in small-lstm big-lstm
    do
	python train_text.py -p ${predictor} -g -0 -b 2 -i 2 -C ${cache}
	python train_text.py -p ${predictor} -g -0 -b 2 -i 2 -C ${cache} -c 
    done
done
