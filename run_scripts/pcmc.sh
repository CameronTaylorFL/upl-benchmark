#! bin/bash 


python3 main.py log=pcmc/incremental/seed$1 model=pcmc dataset=imagenet40-long scenario=incremental seed=$1 \
        model.arch=resnet18 model.layers.layer0.feat_size=512 \
        load_pretrain=True pretrain_log=pcmc//seed1 model.encoder_type=simclr model.sleep_on=True model.pretrained=False \
        scenario.eval_freq=4 model.mem_update=reduce_mem model.update_use=1 model.init_epochs=2 plot=True \
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
        model.layers.layer0.forgetting_factor=4 \
        model.layers.layer0.init_clusters=800 \
        model.layers.layer0.sleep_epochs=2 \
        model.layers.layer0.init_epochs=2 \
        model.layers.layer0.patch_size=60 \
        model.layers.layer0.stride=20 \
        model.n_workers=30