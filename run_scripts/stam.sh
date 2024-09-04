#! bin/bash

python main.py log=stam/blurred/seed1 model=stam dataset=imagenet40-long scenario=incremental seed=1 \
    scenario.eval_freq=4 plot=False dataset.t0_bs=1 \
    dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=0.5 \
    model.layers.layer0.delta=1200 model.layers.layer1.delta=800 model.layers.layer2.delta=400 \
    model.layers.layer0.patch_size=40 model.layers.layer1.patch_size=60 model.layers.layer2.patch_size=90 \
    model.layers.layer0.stride=20 model.layers.layer1.stride=30 model.layers.layer2.stride=30 \
