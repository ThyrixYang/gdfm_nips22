# Generalized Delayed Feedback Model with Post-Click Information in Recommender Systems

## Reproduce results in the paper

Replace seed in config.py to run multiple times and calculate the mean and standard variance.

### Criteo Dataset

Replace the data path in criteo_data.py
```
_local_path = "/path/to/data.txt"
```

#### Pretrain

Note that a checkpoint of pretrained model is provided in ./pretrain_model
If you want to train a new one, 
delete the checkpoint then run with params_name = "criteo_pretrain"

Set params_name = "criteo_pretrain" in main_criteo.py

{'dataset': 'criteo', 'method': 'pretrain', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 2, 'd_nt': 1, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'log_steps': 500, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category', 'current_seed': 0}
MetricAccumulator:
acc: 0.821753
auc: 0.815830
prauc: 0.610097
ll: 0.412973
mce: 0.036591
ece: 0.011126

#### Vanilla (Cross-Entropy, CE)

Set params_name = "criteo_ce" in main_criteo.py

{'dataset': 'criteo', 'method': 'ce', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 2, 'log_steps': 500, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category', 'current_seed': 0}
MetricAccumulator:
acc: 0.822805
auc: 0.820661
prauc: 0.615451
ll: 0.409003
mce: 0.042475
ece: 0.012404

#### Oracle

Set params_name = "criteo_oracle" in main_criteo.py

{'dataset': 'criteo', 'method': 'oracle', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 2, 'log_steps': 500, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category', 'current_seed': 0}
MetricAccumulator:
acc: 0.829567
auc: 0.841372
prauc: 0.642293
ll: 0.389459
mce: 0.039867
ece: 0.008993

#### GDFM

{'dataset': 'criteo', 'method': 'gdfm', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 2, 'd_nt': 7, 'log_steps': 500, 'y_reg_weight': 0.01, 'nt': [0, 360, 900, 3600, 86400, 604800, 2592000], 'alpha': 2, 'beta': 1, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category', 'current_seed': 0}
MetricAccumulator:
acc: 0.826368
auc: 0.834982
prauc: 0.631021
ll: 0.396029
mce: 0.043117
ece: 0.008507

### Taobao Dataset

Replace the data path in taobao_data.py
```
_local_path = "/path/to/data.txt"
```

#### Pretrain

Note that a checkpoint of pretrained model is provided in ./pretrain_model
If you want to train a new one, 
delete the checkpoint then run with params_name = "taobao_pretrain"

{'dataset': 'taobao', 'method': 'pretrain', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 6, 'd_nt': 5, 'weight_decay': 1e-05, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'log_steps': 500, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category'}
MetricAccumulator:
acc: 0.981932
auc: 0.701016
prauc: 0.049348
ll: 0.085431
mce: 0.538143
ece: 0.294352


#### Oracle

{'dataset': 'taobao', 'method': 'oracle', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 6, 'd_nt': 5, 'nt': [120, 600, 7200, 86400, 259200], 'alpha': 2, 'beta': 1, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'log_steps': 500, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category'}
MetricAccumulator:
acc: 0.982081
auc: 0.723500
prauc: 0.063134
ll: 0.083188
mce: 0.540411
ece: 0.135927

#### GDFM

{'dataset': 'taobao', 'method': 'gdfm', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 6, 'd_nt': 5, 'nt': [120, 600, 7200, 86400, 259200], 'alpha': 2, 'beta': 1, 'y_reg_weight': 0.01, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'log_steps': 500, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category'}
MetricAccumulator:
acc: 0.982088
auc: 0.719759
prauc: 0.061672
ll: 0.083906
mce: 0.711116
ece: 0.119731

#### Vanilla (Cross-Entropy, CE)

{'dataset': 'taobao', 'method': 'ce', 'hidden_size': 128, 'y_class_num': 2, 'd_size': 6, 'd_nt': 5, 'nt': [120, 600, 7200, 86400, 259200], 'alpha': 2, 'beta': 1, 'device': 'cuda', 'seed': 0, 'num_workers': 16, 'weight_decay': 1e-06, 'log_steps': 500, 'test_batch_size': 20480, 'batch_size': 4096, 'update_steps': 1, 'pretrain_epochs': 1, 'lr': 0.001, 'optimizer': 'Adam', 'd_type': 'category'}
MetricAccumulator:
acc: 0.982075
auc: 0.715640
prauc: 0.060076
ll: 0.083921
mce: 0.421531
ece: 0.146333