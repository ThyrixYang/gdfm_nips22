import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

import numpy as np
from tqdm import tqdm

from data import DelayedFeedbackDecisionDataset, get_stream

SECONDS_A_DAY = 60 * 60 * 24
SECONDS_AN_HOUR = 60 * 60

cate_bin_size = (1000, 10000, 1000)
taobao_emb_size = 32
taobao_x_emb_size = len(cate_bin_size) * taobao_emb_size
taobao_z_size = 128

taobao_dt_ts = [SECONDS_AN_HOUR // 30,  # 2 min
                SECONDS_AN_HOUR // 6,  # 10 min
                SECONDS_AN_HOUR * 2,  # 2 hour
                SECONDS_A_DAY,  # 1 day
                SECONDS_A_DAY*3,  # 3 day
                ]

taobao_dt_emb_size = len(taobao_dt_ts)

def get_taobao_data_np(debug=False):
    _local_path = "/xxx/UserBehavior.csv"
    with open(_local_path, "r") as f:
        df = pd.read_csv(f, sep=",", header=None)
    df.columns = ["user_id", "item_id",
                  "item_category", "behavior_type", "timestamp"]
    print("before drop duplicates {}".format(df.shape))
    df.drop_duplicates(inplace=True, subset=[
                       "user_id", "item_id", "item_category", "behavior_type"], keep="first")
    df = df.groupby("user_id").filter(lambda x: len(x) >= 10)
    print("after drop duplicates {}".format(df.shape))
    click_df = df[df["behavior_type"] == "pv"]
    click_df = click_df.drop(columns=["behavior_type"])
    click_df = click_df.rename(columns={"timestamp": "click_ts"})
    buy_df = df[df["behavior_type"] == "buy"]
    buy_df = buy_df.drop(columns=["behavior_type"])
    buy_df = buy_df.rename(columns={"timestamp": "pay_ts"})
    cart_df = df[df["behavior_type"] == "cart"]
    cart_df = cart_df.drop(columns=["behavior_type"])
    cart_df = cart_df.rename(columns={"timestamp": "cart_ts"})
    fav_df = df[df["behavior_type"] == "fav"]
    fav_df = fav_df.drop(columns=["behavior_type"])
    fav_df = fav_df.rename(columns={"timestamp": "fav_ts"})

    res_df = click_df.merge(buy_df, how="left", on=[
                            "user_id", "item_id", "item_category"])
    res_df = res_df.merge(cart_df, how="left", on=[
                          "user_id", "item_id", "item_category"])
    res_df = res_df.merge(fav_df, how="left", on=[
                          "user_id", "item_id", "item_category"])
    res_df.fillna(value=-1, inplace=True)
    np_res = res_df.to_numpy()

    res = np_res
    click_ts_mask = np.logical_and(
        res[:, 3] >= 1511539200, res[:, 3] < 1512316800)
    res = res[click_ts_mask]  # some of click time stamp is corrupt
    click_ts, pay_ts, cart_ts, fav_ts = res[:, 3], res[:, 4], res[:, 5], res[:, 6]
    # pay time must be latter than click
    pay_ts_mask = np.logical_and(pay_ts <= click_ts + SECONDS_AN_HOUR // 6, pay_ts >= 0)
    pay_ts[pay_ts_mask] = click_ts[pay_ts_mask] + SECONDS_AN_HOUR*2
    paid_mask = pay_ts >= 0
    # Since the delay of pay is small due to time window used in this dataset
    # We enlarge the pay delay to mimic real delay time
    pay_ts[paid_mask] = (pay_ts[paid_mask] - click_ts[paid_mask])*6 + click_ts[paid_mask]
    too_large_mask = (pay_ts - click_ts) > SECONDS_A_DAY*3
    # set \detla_y = 3 days
    pay_ts[too_large_mask] = click_ts[too_large_mask] + SECONDS_A_DAY*3
    cart_ts_mask = np.logical_and(cart_ts < click_ts, cart_ts >= 0)
    cart_ts[cart_ts_mask] = click_ts[cart_ts_mask] + SECONDS_AN_HOUR // 6
    # fav time must be latter than click (note that some fav occurred before this click, reset to latter)
    fav_ts_mask = np.logical_and(fav_ts < click_ts, fav_ts >= 0)
    fav_ts[fav_ts_mask] = click_ts[fav_ts_mask] + SECONDS_AN_HOUR // 30
    res[:, 4] = pay_ts
    res[:, 5] = cart_ts
    res[:, 6] = fav_ts
    if debug:
        selector = np.random.uniform(0, 1, res.shape[0]) < 0.2
        res = res[selector]
    user_hist = {}
    res = res[res[:, 3].argsort()]
    user_hist_np = []
    print("start cal hist")
    # cnt = 0
    for i in tqdm(range(res.shape[0])):
        user_id = res[i, 0]
        if user_id not in user_hist:
            # item_id, item_cate, action_type
            user_hist[user_id] = [(0, 0, 0) for _ in range(5)]
        this_user_hist = np.array(user_hist[user_id]).reshape(-1)
        user_hist_np.append(this_user_hist)
        if res[i, 4] > 0: # pay
            user_hist[user_id].append((res[i, 1] % cate_bin_size[1], res[i, 2] % cate_bin_size[2], 1))
        elif res[i, 5] > 0: # cart
            user_hist[user_id].append((res[i, 1] % cate_bin_size[1], res[i, 2] % cate_bin_size[2], 2))
        else: # fav
            user_hist[user_id].append((res[i, 1] % cate_bin_size[1], res[i, 2] % cate_bin_size[2], 3))
        user_hist[user_id].pop(0)
    user_hist_np = np.stack(user_hist_np)
    
    res[:, 0] = res[:, 0] % cate_bin_size[0]
    res[:, 1] = res[:, 1] % cate_bin_size[1]
    res[:, 2] = res[:, 2] % cate_bin_size[2]
    res_dict = {
        "x": np.concatenate((res[:, :3], user_hist_np), axis=1),
        "click_ts": res[:, 3],
        "pay_ts": res[:, 4],
        "cart_ts": res[:, 5],
        "fav_ts": res[:, 6],
    }
    return res_dict


class TaobaoData(DelayedFeedbackDecisionDataset):

    def __init__(self, debug=False):
        self.name = "taobao"
        data_dict = get_taobao_data_np(debug)
        x = data_dict["x"]
        click_ts, pay_ts, cart_ts, fav_ts = \
            data_dict["click_ts"], data_dict["pay_ts"], data_dict["cart_ts"], data_dict["fav_ts"]

        ts_x = click_ts
        pay_mask = (pay_ts < 0).astype(np.int64)
        cart_mask = (cart_ts < 0).astype(np.int64)
        fav_mask = (fav_ts < 0).astype(np.int64)
        ts_y = (pay_ts * (1 - pay_mask) + pay_mask *
                (click_ts + SECONDS_A_DAY)).astype(np.int64)

        d_value = []
        ts_d = []
        for _ts in taobao_dt_ts:
            _d_value_pay = (1 - pay_mask) * \
                (ts_y <= ts_x + _ts).astype(np.int64)
            _d_value_cart = (1 - cart_mask) * (cart_ts <=
                                               click_ts + _ts).astype(np.int64)
            _d_value_fav = (1 - fav_mask) * (fav_ts <=
                                             click_ts + _ts).astype(np.int64)
            _d_value = np.stack(
                [_d_value_pay, _d_value_cart, _d_value_fav], axis=1)
            d_value.append(_d_value.reshape((-1, 1, 3)))
            ts_d.append((ts_x + _ts).reshape((-1, 1)))
        d_value = np.concatenate(d_value, axis=1)
        ts_d = np.concatenate(ts_d, axis=1)
        y_value = (pay_ts >= 0).astype(np.int64).reshape((-1))
        ts_seg = SECONDS_AN_HOUR // 3
        x = x.astype(np.int64)
        __start_ts = 1511539200
        split_ts = __start_ts + SECONDS_A_DAY * 2
        super().__init__(ts_x.astype(np.int64), ts_d.astype(np.int64), ts_y.astype(np.int64),
                         x, d_value, y_value, split_ts, ts_seg)

if __name__ == "__main__":
    get_taobao_data_np()
