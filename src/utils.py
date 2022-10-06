from datetime import datetime, timedelta
import time

import torch
import numpy as np

from data import get_stream
from criteo_data import CriteoData, criteo_dt_ts, SECONDS_A_DAY, SECONDS_AN_HOUR
from taobao_data import TaobaoData, taobao_dt_ts
from model import MLP, CriteoMLP


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)

class MovingAverage:
    
    def __init__(self, default_value=0):
        self.count = 0
        self.value = 0
        self.avg = default_value
    
    def add(self, value, batch_size=1):
        if value is None:
            return
        self.value += batch_size * value
        self.count += batch_size
        self.avg = self.value / self.count
    
    def reset(self):
        self.count = 0
        self.value = 0
        self.avg = 0 
        
    def __str__(self):
        return "{:.6f}".format(self.avg)

def get_optimizer(name, params):
    if name == "Adam":
        return lambda model_params: torch.optim.Adam(model_params, lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        raise ValueError("Unknown optimizer name: {}".format(name))

def get_data(dataset_name):
    if dataset_name == "criteo":
        dataset = CriteoData(decision_type="reveal_y", split_ts=10*SECONDS_A_DAY)
        dataset.ts_y = np.ones_like(dataset.ts_y) * SECONDS_A_DAY*30 + dataset.ts_x 
        # reset all y time stamp, GDFM assume that all ts_y is equal. 
        # Intermidiate y information is given by d.
        dataset.reveal(is_test=True)
        elapse_time = criteo_dt_ts
        for e in elapse_time:
            dataset.reveal(dataset.ts_x + e)
        pretrain_dataset, stream_dataset = dataset.split()
        x, y, d, ts, y_mask, d_mask, test_mask = \
            stream_dataset["x"], stream_dataset["y"], stream_dataset["d"], stream_dataset["ts"], \
                stream_dataset["y_mask"], stream_dataset["d_mask"], stream_dataset["test_mask"]
        stream = get_stream(stream_dataset, 
                            dataset.ts_seg)
        return {
            "pretrain_dataset": pretrain_dataset,
            "stream": stream
        }
    elif dataset_name == "taobao":
        dataset = TaobaoData(debug=False)
        dataset.ts_y = np.ones_like(dataset.ts_y) * SECONDS_A_DAY*3 + dataset.ts_x 
        # reset all y time stamp, GDFM assume that all ts_y is equal. 
        # Intermidiate y information is given by d.
        dataset.reveal(is_test=True)
        elapse_time = taobao_dt_ts
        for e in elapse_time:
            dataset.reveal(dataset.ts_x + e)
        pretrain_dataset, stream_dataset = dataset.split()
        x, y, d, ts, y_mask, d_mask, test_mask = \
            stream_dataset["x"], stream_dataset["y"], stream_dataset["d"], stream_dataset["ts"], \
                stream_dataset["y_mask"], stream_dataset["d_mask"], stream_dataset["test_mask"]
        stream = get_stream(stream_dataset, 
                            dataset.ts_seg)
        return {
            "pretrain_dataset": pretrain_dataset,
            "stream": stream
        }
    else:
        raise NotImplementedError()
    
def set_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

class ProgressInfo:
    
    def __init__(self, total_step=None, prefix="", log_steps=None):
        self.prefix = prefix
        self.start_dt = datetime.now()
        self.total_step = total_step
        self._step = 0
        self.log_steps = log_steps
        
    def print_progress(self, msg=""):
        dt = datetime.now() - self.start_dt
        if self.total_step is None:
            print("{}, step:{}, time:{}, {}".format(self.prefix, self._step, str(dt).split(".")[0], msg))
        else:
            if self.step == 0:
                step = 1
            else:
                step = self._step
            percent = "{:03.2f}".format(self._step / self.total_step * 100)
            expected_finish_time = dt / step * self.total_step
            print("{}, step:{}/{} [{}%], time:{}/{}, {}".format(self.prefix, 
                                                               self._step, self.total_step, percent,
                                                          str(dt).split(".")[0], 
                                                          str(expected_finish_time).split(".")[0], msg))
            
    def on_log_steps(self):
        if self.log_steps is not None:
            return self._step % self.log_steps == 0
        return False
            
    def step(self, step_count=1, msg=""):
        self._step += step_count
        if self.on_log_steps():
            self.print_progress(msg)
        