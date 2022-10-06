import bisect

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from data import DelayedFeedbackDecisionDataset, get_stream

_local_path = "/xxx/data.txt"

SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)
criteo_emb_size = 32
criteo_x_emb_size = (len(cate_bin_size) + len(num_bin_size)) * criteo_emb_size
criteo_z_size = 128

criteo_dt_ts = [0,
                SECONDS_AN_HOUR // 10,  # 6 min
                SECONDS_AN_HOUR // 4, # 15 min
                SECONDS_AN_HOUR, # 1 hour
                SECONDS_AN_HOUR * 24, # 1 day
                SECONDS_A_DAY * 7, # 1 week
                SECONDS_A_DAY * 30, # 1 month
                ]

criteo_dt_emb_size = len(criteo_dt_ts)


def get_criteo_data_df():
    df = pd.read_csv(_local_path, sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    df = df[df.columns[2:]]
    for c in df.columns[8:]:
        df[c] = df[c].fillna("")
        df[c] = df[c].astype(str)
    for c in df.columns[:8]:
        df[c] = df[c].fillna(-1)
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.columns = [str(i) for i in range(17)]
    df.reset_index(inplace=True)
    res = {
        "df": df,
        "click_ts": click_ts,
        "pay_ts": pay_ts
    }
    return res

def load_criteo_data():
    res = get_criteo_data_df()
    df, click_ts, pay_ts = res["df"], res["click_ts"], res["pay_ts"]
    data = []
    for i in range(8, 17):
        c = str(i)
        hash_value = pd.util.hash_array(df[c].to_numpy()) % cate_bin_size[i-8]
        data.append(hash_value.reshape(-1, 1))
    for i in range(8):
        c = str(i)
        labels = list(range(num_bin_size[i]))
        out, bins = pd.cut(df[c], bins=num_bin_size[i], retbins=True, labels=labels)
        data.append(out.to_numpy().reshape((-1, 1)))
    data = np.concatenate(data, axis=1)
    res = {
        "x": data,
        "click_ts": click_ts,
        "pay_ts": pay_ts
    }
    return res


class CriteoData(DelayedFeedbackDecisionDataset):
    
    def __init__(self,
                 split_ts,
                 decision_type="reveal_y"):
        assert decision_type in ["reveal_y"]
        self.name = "criteo"
        criteo_data = load_criteo_data()
        x, click_ts, pay_ts = criteo_data["x"], criteo_data["click_ts"], criteo_data["pay_ts"]
        ts_x = click_ts
        pay_mask = (pay_ts == -1).astype(np.int64)
        ts_y = (pay_ts * (1 - pay_mask) + pay_mask * (click_ts + SECONDS_A_DAY * 30)).astype(np.int64)
        if decision_type == "reveal_y":
            d_value = []
            ts_d = []
            for _ts in criteo_dt_ts:
                _d_value = (1 - pay_mask) * (ts_y <= ts_x + _ts).astype(np.int64)
                d_value.append(_d_value.reshape((-1, 1, 1)))
                ts_d.append((ts_x + _ts).reshape((-1, 1)))
            d_value = np.concatenate(d_value, axis=1)
            ts_d = np.concatenate(ts_d, axis=1)
        else:
            raise ValueError("Unknown decision type: {}".format(decision_type))
        y_value = (pay_ts != -1).astype(np.int64).reshape((-1))
        ts_seg = SECONDS_AN_HOUR
        x = x.astype(np.int64)
        super().__init__(ts_x, ts_d, ts_y, x, d_value, y_value, split_ts, ts_seg)

if __name__ == "__main__":
    data = CriteoData(split_ts=SECONDS_A_DAY*10)