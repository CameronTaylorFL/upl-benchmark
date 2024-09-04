import yaml, json

import neptune
import pickle as p
import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from core.utils import *
from core.stream.streams import load_stream
from core.models.stam.STAM import STAM
from core.models.pcmc.pcmc import PCMC
from core.models.pcmc_whole.pcmc_whole import PCMC_Whole
from core.models.scale.scale_new import SCALE
from core.models.pca.pca import PCAModel
from core.models.whole_baseline.whole_baseline_2 import WholeBaseline
from core.eval.eval import evaluate

import matplotlib.pyplot as plt
import torchvision.utils as vutils 

def set_seed(seed=1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def cornerstone_plot(acc, novel, past, sleep_times, task_bounds, config, mode):
    plt.figure(figsize=(9, 3), dpi=600)

    plt.plot(np.arange(len(acc)) + 0.5, acc, '-o', markersize=3, label='Overall', color='green')
    if novel != None:
        plt.plot(np.arange(1, len(novel)+1) + 0.5, novel,  '-o', markersize=3, label='Novel Classes', color='red', alpha=0.7)
        plt.plot(np.arange(1, len(past)+1) + 0.5, past, '-o', markersize=3, label='Past Classes', color='blue', alpha=0.7)
        
    if len(sleep_times) > 0:
        plt.vlines(sleep_times + 1, 0, 100, label='Sleep', color='gray', linestyles='dotted', linewidth=2)

    plt.vlines([1], 0, 100, color='black', linestyles='dashed', label='Stream Begins', linewidth=3)

    #plt.yticks(np.arange(30, 70, 10), np.arange(30, 70, 10))
    plt.ylabel('Classification Acc (%)')

    plt.xticks(task_bounds, labels=[f'T{i}' for i in range(len(task_bounds))])
    plt.xlabel('Stream Progression')

    plt.title('STEAM Performance')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.savefig(smart_dir(f'logs/{config.log}/{config.dataset.name}') + f'cornerstone_{mode}.png')
    plt.close()


@hydra.main(version_base=None, config_path='config', config_name='main')
def main(config: DictConfig) -> None:

    set_seed(config.seed)
    with open(smart_dir(f'logs/{config.log}/{config.dataset.name}') + "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    stream = load_stream(config)
    print(config.model.name)

    if config.model.name == 'stam':
        model = STAM(config)
    if config.model.name == 'pcmc':
        model = PCMC(config)
    if config.model.name == 'pcmc-whole':
        model = PCMC_Whole(config)
    if config.model.name == 'scale':
        model = SCALE(config)
    if config.model.name == 'whole-baseline':
        model = WholeBaseline(config)
    if config.model.name == 'pca':
        model = PCAModel(config)


    class_perf = []
    class_pc = []

    clust_perf = []
    clust_pc = []

    task_bounds = stream.task_bounds(config.scenario.eval_freq)
    eval_times, sep = stream.eval_times(config.scenario.eval_freq)
    print(eval_times, sep)
    
    model.pretrain(stream.pretrain_dataloader)
    sup, evl = stream.eval_loaders(0)
    
    print('Starting T0 Eval')
    class_acc, class_pc_acc, clust_acc, clust_pc_acc = model.eval(sup, evl, 0, 0)
    print('Done T0 Eval')
    class_perf.append(class_acc)
    class_pc.append(class_pc_acc)

    clust_perf.append(clust_acc)
    clust_pc.append(clust_pc_acc)
    
    sleep_times = []

    #model.plots()
    cornerstone_plot(class_perf, None, None, sleep_times, task_bounds, config, 'class')
    cornerstone_plot(clust_perf, None, None, sleep_times, task_bounds, config, 'clust')
    #print('Starting Stream')
    print('STREAM BS: ', config.dataset.stream_bs)
    print('STREAM LENGTH: ', len(stream))

    cur_eval = 0
    for it, (data, _, t) in enumerate(tqdm(stream)):
        model(data, t)

        # Temporary
        if config.model.name == 'pcmc':
            if (it == config.model.sleep_start) or (it - config.model.sleep_start) % config.model.sleep_freq == 0:
                print(f'adding sleep time - {it}')
                sleep_times.append(it)
                print('Sleep TIMES: ', sleep_times)
        elif config.model.name == 'whole-baseline':
            if (it + (config.model.sleep_freq // 2)) % config.model.sleep_freq == 0:
                sleep_times.append(it)
                
        if int(it * config.dataset.stream_bs) in eval_times:
            print(f'Eval Task: {t} - Iter: {it}')
            sup, evl = stream.eval_loaders(t)
            class_acc, class_pc_acc, clust_acc, clust_pc_acc = model.eval(sup, evl, t, it)

            class_perf.append(class_acc)
            class_pc.append(class_pc_acc)

            clust_perf.append(clust_acc)
            clust_pc.append(clust_pc_acc)

            #model.plots()
            novel = []
            past = []
            for results in class_pc[1:]:
                novel.append(np.mean(np.array(results)[-config.dataset.task_sizes[t]:]))
                past.append(np.mean(np.array(results)[:-config.dataset.task_sizes[t]]))

            
            cornerstone_plot(class_perf, 
                             novel, 
                             past, 
                             np.array(sleep_times) // sep,
                             np.array(task_bounds),
                             config, 'class')
            novel = []
            past = []
            for results in clust_pc[1:]:
                novel.append(np.mean(np.array(results)[-config.dataset.task_sizes[t]:]))
                past.append(np.mean(np.array(results)[:-config.dataset.task_sizes[t]]))

            cornerstone_plot(clust_perf, 
                             novel, 
                             past, 
                             np.array(sleep_times) // sep,
                             np.array(task_bounds),
                             config, 'clust')

            cur_eval += 1
    
    print('Done')
    
    results = {'class_acc' : class_perf, 
               'class_pc' : class_pc,
               'clust_acc' : clust_perf,
               'clust_pc' : clust_pc,
               'sleep_times' : sleep_times,
               'task_bounds' : task_bounds}

    with open(smart_dir(f'logs/{config.log}/{config.dataset.name}') + "results.pkl", "wb") as f:
        p.dump(results, f)
    
    #print(results)
    #run.stop()



if __name__ == '__main__':
    main()