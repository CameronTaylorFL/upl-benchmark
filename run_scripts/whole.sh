#! bin/bash

python main.py log=whole/incremental/seed$1 model=whole-baseline dataset=imagenet40-long scenario=incremental seed=$1 \
    model.sleep_epochs=250 model.init_epochs=300 model.load_pretrain=False pretrain_log=whole/incremental/seed$1 model.pretrained=True \
    dataset.super_size=200 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
    dataset.stream_bs=1 scenario.eval_freq=2 model.sleep_freq=1000 model.k_scale=2 model.init_size=2500 \
    model.stm_size=225 model.ltm_size=175
