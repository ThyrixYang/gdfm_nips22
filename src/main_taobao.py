import time

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, set_seed
from metric import metric_mean_and_std
import config
from solver_taobao import Solver


def get_solver(params):
    assert "taobao" in params["dataset"]
    dataset = get_data(params["dataset"])
    pretrain_dataset, stream_dataset = dataset["pretrain_dataset"], dataset["stream"]
    return Solver(pretrain_dataset,
                  stream_dataset,
                  params), dataset


def main():
    params_name = "taobao_ce"
    params = config.experiment_params[params_name]
    if isinstance(params["seed"], int):
        seed_list = [params["seed"]]
    else:
        seed_list = params["seed"]
    print("seed_list:", seed_list)
    metric_list = []
    for seed in seed_list:
        set_seed(seed)
        print(params)
        solver, dataset = get_solver(params)
        print("pretraining")
        solver.pretrain()
        print("stream training and testing")
        results = solver.stream_train_and_predict()

        metric = results["metric"]
        print(params)
        print(metric)
        metric_list.append(metric)
    metric_mean_and_std(metric_list)


if __name__ == "__main__":
    main()
    print("finished")
