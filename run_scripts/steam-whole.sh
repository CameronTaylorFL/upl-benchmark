#! bin/bash 

python3 main.py log=steam-whole/main-results/seed1 model=stam2-whole dataset=imagenet40-long scenario=incremental seed=1 \
        load_pretrain=False pretrain_log=steam-whole/test model.encoder_type=simclr model.sleep_on=True \
        scenario.eval_freq=4 model.mem_update=none model.update_use=1 model.init_epochs=300 plot=False \
        dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
        model.sleep_start=1000 model.sleep_freq=2000 \
        model.layers.layer0.delta=200 \
        model.layers.layer0.temperature=0.5 \
        model.layers.layer0.wd=1e-4 \
        model.layers.layer0.sleep_bs=512 \
        model.layers.layer0.beta=0.95 \
        model.layers.layer0.theta=15 \
        model.layers.layer0.rho=0.25 \
        model.layers.layer0.alpha=0.0 \
        model.layers.layer0.M=15 \
        model.layers.layer0.init_clusters=100 \
        model.layers.layer0.sleep_epochs=200 \
        model.layers.layer0.init_epochs=300 \
        model.layers.layer0.patch_size=120 \
        model.layers.layer0.stride=1 \
        model.n_workers=15

python3 main.py log=steam-whole/main-results/seed2 model=stam2-whole dataset=imagenet40-long scenario=incremental seed=2 \
        load_pretrain=False pretrain_log=steam-whole/test model.encoder_type=simclr model.sleep_on=True \
        scenario.eval_freq=4 model.mem_update=none model.update_use=1 model.init_epochs=300 plot=False \
        dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
        model.sleep_start=1000 model.sleep_freq=2000 \
        model.layers.layer0.delta=200 \
        model.layers.layer0.temperature=0.5 \
        model.layers.layer0.wd=1e-4 \
        model.layers.layer0.sleep_bs=512 \
        model.layers.layer0.beta=0.95 \
        model.layers.layer0.theta=15 \
        model.layers.layer0.rho=0.25 \
        model.layers.layer0.alpha=0.0 \
        model.layers.layer0.M=15 \
        model.layers.layer0.init_clusters=100 \
        model.layers.layer0.sleep_epochs=200 \
        model.layers.layer0.init_epochs=300 \
        model.layers.layer0.patch_size=120 \
        model.layers.layer0.stride=1 \
        model.n_workers=15

python3 main.py log=steam-whole/main-results/seed3 model=stam2-whole dataset=imagenet40-long scenario=incremental seed=3 \
        load_pretrain=False pretrain_log=steam-whole/test model.encoder_type=simclr model.sleep_on=True \
        scenario.eval_freq=4 model.mem_update=none model.update_use=1 model.init_epochs=300 plot=False \
        dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0 \
        model.sleep_start=1000 model.sleep_freq=2000 \
        model.layers.layer0.delta=200 \
        model.layers.layer0.temperature=0.5 \
        model.layers.layer0.wd=1e-4 \
        model.layers.layer0.sleep_bs=512 \
        model.layers.layer0.beta=0.95 \
        model.layers.layer0.theta=15 \
        model.layers.layer0.rho=0.25 \
        model.layers.layer0.alpha=0.0 \
        model.layers.layer0.M=15 \
        model.layers.layer0.init_clusters=100 \
        model.layers.layer0.sleep_epochs=200 \
        model.layers.layer0.init_epochs=300 \
        model.layers.layer0.patch_size=120 \
        model.layers.layer0.stride=1 \
        model.n_workers=15