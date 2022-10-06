import bisect
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

class DelayedFeedbackDecisionDataset:

    def __init__(self,
                 ts_x,
                 ts_d,
                 ts_y,
                 x_value,
                 d_value,
                 y_value,
                 split_ts,
                 ts_seg):
        """_summary_
        N = number of samples
        nt = number of decision timestamp
        dx = dimension of x
        dd = dimension of d
        dy = dimension of y
        :param ts_x: shape=(N,)
        :param ts_d: shape=(N, nt)
        :param ts_y: shape=(N,)
        :param x_value: shape=(N, dx)
        :param d_value: shape=(N, nt, dd)
        :param y_value: shape=(N, dy)
        :param batch_len: int, in timestamp
        :param deadline: int
        """
        assert len(ts_x.shape) == 1
        assert len(ts_y.shape) == 1
        assert len(ts_d.shape) == 2
        assert len(x_value.shape) == 2
        assert len(d_value.shape) == 3
        assert len(y_value.shape) == 1
        self.ts_x = ts_x
        self.ts_d = ts_d
        self.ts_y = ts_y
        self.x_value = x_value
        self.d_value = d_value
        self.y_value = y_value
        self.split_ts = split_ts
        self.ts_seg = ts_seg
        self.N, self.nt, self.dd = self.d_value.shape
        _, self.dx = self.x_value.shape

        self.data_x = None
        
    def reveal(self, 
               reveal_ts=None, 
               is_test=False):
        if is_test:
            reveal_ts = self.ts_x - 1
        assert reveal_ts is not None
        assert isinstance(reveal_ts, int) or reveal_ts.shape[0] == self.ts_x.shape[0]
        assert len(reveal_ts.shape) == 1
        
        d_mask = np.zeros((self.N, self.nt), dtype=np.bool)
        if is_test:
            test_mask = np.ones((self.N,), dtype=np.bool)
        else:
            test_mask = np.zeros((self.N,), dtype=np.bool)
        y_mask = self.ts_y <= reveal_ts
        
        for i in range(self.nt):
            d_mask[:, i] = (self.ts_d[:, i] <= reveal_ts)
            
        if self.data_x is None:
            self.data_x = []
            self.data_y = []
            self.data_d = []
            self.data_ts = []
            self.data_x_ts = []
            self.data_y_mask = []
            self.data_test_mask = []
            self.data_d_mask = []
            
        self.data_x.append(self.x_value)
        self.data_y.append(self.y_value)
        self.data_d.append(self.d_value)
        self.data_ts.append(reveal_ts)
        self.data_x_ts.append(self.ts_x)
        self.data_y_mask.append(y_mask)
        self.data_test_mask.append(test_mask)
        self.data_d_mask.append(d_mask)
            
        
    def split(self):
        self.data_x = np.concatenate(self.data_x, axis=0)
        self.data_y = np.concatenate(self.data_y, axis=0)
        self.data_d = np.concatenate(self.data_d, axis=0)
        self.data_ts = np.concatenate(self.data_ts, axis=0)
        self.data_x_ts = np.concatenate(self.data_x_ts, axis=0)
        self.data_y_mask = np.concatenate(self.data_y_mask, axis=0)
        self.data_d_mask = np.concatenate(self.data_d_mask, axis=0)
        self.data_test_mask = np.concatenate(self.data_test_mask, axis=0)
        self.sort()
        split_index = bisect.bisect_left(self.data_ts, self.split_ts)
        # Get all the ground truth label, so we need to put real negatives to the pretrain dataset
        # Note that this will not cause label leakage since all these data are not in the testing set in the streaming testing
        # This is important since the ground truth label of Criteo dataset is 30 days later, 
        # if we do not do this there will be no label left
        pretrain_mask = self.data_x_ts <= self.split_ts
        stream_mask = self.data_x_ts > self.split_ts
        left = {
            "x": self.data_x[pretrain_mask],
            "y": self.data_y[pretrain_mask],
            "d": self.data_d[pretrain_mask],
            "ts": self.data_ts[pretrain_mask],
            "y_mask": self.data_y_mask[pretrain_mask],
            "d_mask": self.data_d_mask[pretrain_mask],
            "test_mask": self.data_test_mask[pretrain_mask],
        }
        right = {
            "x": self.data_x[split_index:],
            "y": self.data_y[split_index:],
            "d": self.data_d[split_index:],
            "ts": self.data_ts[split_index:],
            "y_mask": self.data_y_mask[split_index:],
            "d_mask": self.data_d_mask[split_index:],
            "test_mask": self.data_test_mask[split_index:]
        }
        return left, right

    def sort(self):
        idx = list(range(len(self.data_ts)))
        idx = sorted(idx, key=lambda i: self.data_ts[i])
        self.data_x = self.data_x[idx]
        self.data_y = self.data_y[idx]
        self.data_d = self.data_d[idx]
        self.data_ts = self.data_ts[idx]
        self.data_x_ts = self.data_x_ts[idx]
        self.data_test_mask = self.data_test_mask[idx]
        self.data_y_mask = self.data_y_mask[idx]
        self.data_d_mask = self.data_d_mask[idx]
        
def get_test_data_from_batch(x, y, d, y_mask, d_mask, test_mask):
    return (torch.from_numpy(x[test_mask]), 
            torch.from_numpy(y[test_mask]), 
            torch.from_numpy(d[test_mask]), 
            torch.from_numpy(y_mask[test_mask]),
            torch.from_numpy(d_mask[test_mask]))

def get_train_data_from_batch(x, y, d, y_mask, d_mask, test_mask):
    mask = np.logical_not(test_mask)
    return (torch.from_numpy(x[mask]), 
            torch.from_numpy(y[mask]), 
            torch.from_numpy(d[mask]), 
            torch.from_numpy(y_mask[mask]),
            torch.from_numpy(d_mask[mask]))

def get_stream(data, ts_seg, debug=False):
    max_ts = np.max(data["ts"])
    min_ts = np.min(data["ts"])
    stream = []
    print("getting stream data")
    cnt = 0
    for _ts_start in tqdm(range(min_ts, max_ts + 2, ts_seg)):
        cnt += 1
        if cnt > 100 and debug:
            break
        _ts_end = _ts_start + ts_seg
        start_idx = bisect.bisect_left(data["ts"], _ts_start) # a[start_idx] >= _ts_start
        end_idx = bisect.bisect_left(data["ts"], _ts_end) # a[end_idx - 1] < _ts_end
        if end_idx > start_idx:
            batch_x = data["x"][start_idx:end_idx]
            batch_y = data["y"][start_idx:end_idx]
            batch_d = data["d"][start_idx:end_idx]
            batch_y_mask = data["y_mask"][start_idx:end_idx]
            batch_d_mask = data["d_mask"][start_idx:end_idx]
            batch_test_mask = data["test_mask"][start_idx:end_idx]
            test = get_test_data_from_batch(batch_x, batch_y, batch_d, batch_y_mask, batch_d_mask, batch_test_mask)
            train = get_train_data_from_batch(batch_x, batch_y, batch_d, batch_y_mask, batch_d_mask, batch_test_mask)
            stream.append((train, test))
    return stream