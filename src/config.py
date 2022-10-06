from criteo_data import criteo_emb_size, criteo_x_emb_size, criteo_z_size, criteo_dt_emb_size, SECONDS_AN_HOUR, SECONDS_A_DAY
from criteo_data import criteo_dt_ts
from taobao_data import taobao_emb_size, taobao_x_emb_size, taobao_z_size, taobao_dt_emb_size, taobao_dt_ts


global_config = {
    "device": "cuda",
    "seed": 0,
    "num_workers": 16,
    "weight_decay": 1e-6,
    "log_steps": 10,
    "test_batch_size": 20480,
    "batch_size": 4096,
    "update_steps": 1,
    "pretrain_epochs": 1,
    "log_steps": 500,
    "lr": 1e-3,
    "optimizer": "Adam",
    "d_type": "category",
}

experiment_params = {
    "criteo_pretrain": {
        "dataset": "criteo",
        "method": "pretrain",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
        "d_nt": 1,
    },
    "taobao_pretrain": {
        "dataset": "taobao",
        "method": "pretrain",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "weight_decay": 1e-5,
    },
    "taobao_gdfm": {
        "dataset": "taobao",
        "method": "gdfm",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "alpha": 2,
        "beta": 1,
        "y_reg_weight": 0.01,
    },
    "taobao_oracle": {
        "dataset": "taobao",
        "method": "oracle",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "alpha": 2,
        "beta": 1,
    },
    "taobao_ce": {
        "dataset": "taobao",
        "method": "ce",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 6,
        "d_nt": len(taobao_dt_ts),
        "nt": taobao_dt_ts,
        "alpha": 2,
        "beta": 1,
    },
    "criteo_gdfm": {
        "dataset": "criteo",
        "method": "gdfm",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
        "d_nt": 7,
        "y_reg_weight": 0.01,
        "nt": criteo_dt_ts,
        "alpha": 2,
        "beta": 1,
    },
    "criteo_ce": {
        "dataset": "criteo",
        "method": "ce",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
    },
    "criteo_oracle": {
        "dataset": "criteo",
        "method": "oracle",
        "hidden_size": 128,
        "y_class_num": 2,
        "d_size": 2,
    },
}

for k in experiment_params.keys():
    for _k in global_config.keys():
        if _k not in experiment_params[k]:
            experiment_params[k][_k] = global_config[_k]
