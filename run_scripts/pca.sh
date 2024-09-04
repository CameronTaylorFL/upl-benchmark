#!/bin/sh
python main.py log=stam2/pretrained/seed1 model=pca dataset=stream51-28 scenario=dynamic seed=1 \
    dataset.stream_bs=1 scenario.eval_freq=4 \
    model.k_scale=2 model.components=100

python main.py log=stam2/pretrained/seed2 model=pca dataset=stream51-28 scenario=dynamic seed=2 \
    dataset.stream_bs=1 scenario.eval_freq=4 \
    model.k_scale=2 model.components=100

python main.py log=stam2/pretrained/seed3 model=pca dataset=stream51-28 scenario=dynamic seed=3 \
    dataset.stream_bs=1 scenario.eval_freq=4 \
    model.k_scale=2 model.components=100

