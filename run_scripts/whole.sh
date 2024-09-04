#! bin/bash

python main.py log=scale/blurred/seed1 model=scale dataset=imagenet40-long scenario=blurred seed=1 \
    dataset.t0_factor=1.0 \
    model.epochs=1 model.init_epochs=300 model.lr=0.01 model.load_pretrain=True pretrain_log=scale/blurred/seed1 \
    dataset.stream_bs=1 scenario.eval_freq=4 model.k_scale=2 \
    model.stream_train=True model.mem_max_classes=40 model.mem_size=1280

python3 main.py log=pcmc/blurred/seed2 model=stam2-1 dataset=imagenet40-long scenario=blurred seed=2 \
        model.arch=resnet18 model.layers.layer0.feat_size=512 \
        load_pretrain=True pretrain_log=pcmc/seed2 model.encoder_type=simclr model.sleep_on=True model.pretrained=False \
        scenario.eval_freq=4 model.mem_update=reduce_mem model.update_use=1 model.init_epochs=500 plot=False \
        dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
        model.sleep_start=1000 model.sleep_freq=2000 \
        model.layers.layer0.delta=400 \
        model.layers.layer0.lr=0.6 \
        model.layers.layer0.temperature=0.5 \
        model.layers.layer0.wd=1e-4 \
        model.layers.layer0.sleep_bs=512 \
        model.layers.layer0.pretrain_bs=256 \
        model.layers.layer0.beta=0.95 \
        model.layers.layer0.theta=30 \
        model.layers.layer0.rho=0.25 \
        model.layers.layer0.alpha=0.0 \
        model.layers.layer0.M=30 model.layers.layer0.init_M=17 model.layers.layer0.M_min=5 \
        model.layers.layer0.forgetting_factor=1 \
        model.layers.layer0.init_clusters=800 \
        model.layers.layer0.sleep_epochs=250 \
        model.layers.layer0.init_epochs=500 \
        model.layers.layer0.patch_size=60 \
        model.layers.layer0.stride=20 \
        model.n_workers=14


python main.py log=whole/blurred/seed2 model=whole-baseline dataset=imagenet40-long scenario=blurred seed=2 \
    model.sleep_epochs=250 model.init_epochs=500 model.load_pretrain=True pretrain_log=whole/full-base-run model.pretrained=False \
    dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
    dataset.stream_bs=1 scenario.eval_freq=4 model.sleep_freq=2000 model.k_scale=2 model.init_size=3750 \
    model.stm_size=225 model.ltm_size=225


python main.py log=scale/blurred/seed2 model=scale dataset=imagenet40-long scenario=blurred seed=2 \
    dataset.t0_factor=1.0 \
    model.epochs=1 model.init_epochs=300 model.lr=0.01 model.load_pretrain=True pretrain_log=scale/blurred/seed1 \
    dataset.stream_bs=1 scenario.eval_freq=4 model.k_scale=2 \
    model.stream_train=True model.mem_max_classes=40 model.mem_size=1280


python main.py log=stam/blurred/seed2 model=stam dataset=imagenet40-long scenario=blurred seed=2 \
    scenario.eval_freq=4 plot=False dataset.t0_bs=1 \
    dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=0.5 \
    model.layers.layer0.delta=1200 model.layers.layer1.delta=800 model.layers.layer2.delta=400 \
    model.layers.layer0.patch_size=40 model.layers.layer1.patch_size=60 model.layers.layer2.patch_size=90 \
    model.layers.layer0.stride=20 model.layers.layer1.stride=30 model.layers.layer2.stride=30 \