#! bin/bash


python main.py log=scale/blurred/seed1 model=scale dataset=imagenet40-long scenario=blurred seed=1 \
    model.epochs=1 model.init_epochs=300 model.lr=0.01 model.load_pretrain=True pretrain_log=scale/blurred/seed1 \
    dataset.super_size=5 dataset.test_size=5 dataset.stream_size=1000 dataset.t0_factor=1.0 \
    dataset.stream_bs=1 scenario.eval_freq=1 model.k_scale=2 \
    model.stream_train=True model.mem_max_classes=40 model.mem_size=1280
