## Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning
PyTorch code for the COLLAS 2024 paper:\
**Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning**\
**_[Cameron Taylor]_**, Vassilis Vassiliades, [Constantine Dovrolis]\
International Joint Conference on Artificial Intelligence (IJCAI), 2021\


## Requirements
To install Python 3 requirements:

```setup
python3 -m venv .pcmc
source .pcmc/bin/activate
pip install -r requirements.txt
```

## Basic Experiments
The bash scripts to run each model are included in the run_scripts folder. An example run for PCMC on the imagenet40 dataset.

``` 
python3 main.py log=pcmc/$1 model=pcmc dataset=imagenet40-long scenario=incremental seed=$2 \
        model.arch=resnet18 model.layers.layer0.feat_size=512 \
        load_pretrain=True pretrain_log=pcmc/seed$2 model.encoder_type=simclr model.sleep_on=True model.pretrained=False \
        scenario.eval_freq=4 model.mem_update=reduce_mem model.update_use=1 model.init_epochs=300 plot=True \
        dataset.super_size=100 dataset.test_size=100 dataset.stream_size=1000 dataset.t0_factor=1.0

```

## Contributing
MIT License

## Citation
If you found our work useful for your research, please cite our work:

        @article{taylor2024patch,
             title={Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning},
             author={Taylor, Cameron  and Vassiliades, Vassilis  and Dovrolis, Constantine},
             journal={Proceedings of The 3rd Conference on Lifelong Learning Agents},
             year={2024},
        }

[Cameron Taylor]: https://www.linkedin.com/in/cameron-taylor95/
[Constantine Dovrolis]: https://www.cc.gatech.edu/fac/Constantinos.Dovrolis/