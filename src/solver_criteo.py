from collections import OrderedDict
import pickle
from itertools import chain
import os

import numpy as np
import torch
import torch.nn.functional as F

from metric import MetricAccumulator, cal_metric
from utils import ProgressInfo, get_optimizer
from data import get_train_data_from_batch
from model import CriteoMLP, CriteoDXY

class Solver:

    def __init__(self,
                 pretrain_dataset,
                 stream_dataset,
                 params):
        
        self.params = params
        self.device = params["device"]
        self.pretrain_dataset = pretrain_dataset
        self.stream_dataset = stream_dataset
        self.method = params["method"]
        if "debug" in params and params["debug"]:
            self.debug = True
        else:
            self.debug = False

        self.hidden_size = self.params["hidden_size"]
        self.y_class_num = self.params["y_class_num"]
        if self.method == "gdfm":
            self.d_type = self.params["d_type"]
            self.d_size = self.params["d_size"]
            self.d_nt = self.params["d_nt"]
            self.nt = self.params["nt"]
            self.alpha = self.params["alpha"]
            self.beta = self.params["beta"]
            self.p_d_xy = CriteoDXY(y_size=self.y_class_num, d_size=self.d_size*self.d_nt).to(self.device)
            self.optimizer_dxy = get_optimizer(
                params["optimizer"], params)(self.p_d_xy.parameters())
            assert self.d_type in ["category"]
            
        self.p_y_x = CriteoMLP(output_size=self.y_class_num).to(self.device)
        self.reg_p_y_x = CriteoMLP(output_size=self.y_class_num).to(self.device)
        
        for param in self.reg_p_y_x.parameters():
            param.detach_()

        self.optimizer_xy = get_optimizer(
            params["optimizer"], params)(self.p_y_x.parameters())

    def update_step(self, x, y, d, y_mask, d_mask, streaming):
        batch_size = x.shape[0]
        # d_mask.shape = batch_size, nt (14)
        
        if batch_size <= 1:
            # pytorch batchnorm does not work with batch_size=1
            # and using small batch will lead to large variance, so we drop them
            return 0, None

        if self.method != "gdfm":
            if self.method in ["ce"]:
                if not streaming:
                    raise ValueError()
                else:
                    _d_mask = d_mask[:, 6] # by construction, d[6] is accurate label with 30 day delay.
                    _label = d[:, 6]
                self.p_y_x.train()
                labeled_x = x[_d_mask]
                labeled_y = _label[_d_mask]
                if labeled_x.shape[0] <= 1:
                    # pytorch batchnorm does not work with batch_size=1
                    # and using small batch will lead to large variance, so we drop them
                    return None
                logits = self.p_y_x(labeled_x)
                prob = F.softmax(logits, dim=1)
                loss = F.nll_loss(F.log_softmax(
                            logits, dim=1), labeled_y.reshape((-1))
                        )
            elif self.method in ["pretrain"]:
                if streaming:
                    raise ValueError()
                day30_mask = d_mask[:, 6]
                day30_label = d[:, 6]
                labeled_x = x[day30_mask]
                labeled_y = day30_label[day30_mask]
                if labeled_x.shape[0] <= 1:
                    # pytorch batchnorm does not work with batch_size=1
                    # and using small batch will lead to large variance, so we drop them
                    return None
                logits = self.p_y_x(labeled_x)
                prob = F.softmax(logits, dim=1)
                loss = F.nll_loss(F.log_softmax(
                    logits, dim=1), labeled_y.reshape((-1)))
            elif self.method == "oracle":
                if streaming:
                    self.p_y_x.train()
                    # oracle can use label on data arrival
                    h0_mask = torch.logical_and(d_mask[:, 0], torch.logical_not(d_mask[:, 1]))
                    labeled_x = x[h0_mask]
                    labeled_y = y[h0_mask]
                    if labeled_x.shape[0] <= 1:
                        return None
                    logits = self.p_y_x(labeled_x)
                    prob = F.softmax(logits, dim=1)
                    loss = F.nll_loss(F.log_softmax(
                        logits, dim=1), labeled_y.reshape((-1)))
                else:
                    raise ValueError()
            else:
                raise ValueError(
                    "Unknown method type: {}".format(self.method))
                
            self.optimizer_xy.zero_grad()
            loss.backward()
            self.optimizer_xy.step()
            return {
                "loss": loss.item(),
                "target": labeled_y.detach().cpu().numpy(),
                "prob": prob.detach().cpu().numpy(),
                "cvr": torch.mean((labeled_y == 1).float()).item(),
                "real_batch_size": labeled_y.shape[0],
            }

        # pretrain of GDFM
        if not streaming:
            # pretrain p_d_xy, use all ground truth label
            y_onehot = F.one_hot(y, self.y_class_num).float()
            d_pred = self.p_d_xy(x, y_onehot).view((batch_size, self.d_nt, self.d_size))
            if self.d_type == "category":
                _d_pred = d_pred.view((batch_size*self.d_nt, -1))
                _d_label = d.view((batch_size*self.d_nt,))
                d_loss = F.nll_loss(F.log_softmax(_d_pred, dim=1), 
                                    _d_label, 
                                    reduction="none")
                d_loss = torch.mean(d_loss)
            else:
                raise NotImplementedError()
                d_loss = torch.mean(d_loss)

            self.optimizer_dxy.zero_grad()
            d_loss.backward()
            self.optimizer_dxy.step()
            
            return {
                "d_loss": d_loss.item(),
            }
        # stream
        else:
            self.p_d_xy.eval()
            
            # update p_y_x with p_d_xy
            reg_y_logits = self.reg_p_y_x(x).detach()
            
            y_logits = self.p_y_x(x)
            
            y_reg_loss = F.kl_div(
                        F.log_softmax(y_logits, dim=1), 
                        F.softmax(reg_y_logits, dim=1), 
                        reduction="batchmean")
            y_reg_loss = y_reg_loss * self.params["y_reg_weight"]

            sample_labels = torch.arange(0, self.y_class_num, device=x.device)

            sample_labels = sample_labels.unsqueeze(
                0).repeat(batch_size, 1)
            # batch_size, y_class_num

            sample_labels = F.one_hot(sample_labels, self.y_class_num).float()
            # shape= batch_size, y_class_num, y_class_num
            
            sample_x = x.unsqueeze(1).repeat(1, self.y_class_num, 1)
            # shape= batch_size, y_class_num, x_size
            
            sample_d = d.unsqueeze(1).repeat(1, self.y_class_num, 1, 1)
            # shape= batch_size, y_class_num, nt, d_size

            # shape= batch_size*nt*y_class_num, hidden_size + y_class_num
            
            pred_d = self.p_d_xy(
                sample_x.view((batch_size*self.y_class_num,-1)), 
                sample_labels.view((batch_size*self.y_class_num,-1))
                ).view(
                (batch_size, self.y_class_num, self.d_nt, self.d_size))
            # shape= batch_size, y_class_num, nt, d_size
            
            # sample_d.shape= batch_size, y_class_num, nt, d_size=1
            _sample_d_mask = F.one_hot(sample_d.squeeze(3), self.d_size).float()
            # _sample_d_mask.shape= batch_size, y_class_num, nt, d_size
            log_prob_d = torch.sum(F.log_softmax(pred_d, dim=3) * _sample_d_mask, dim=3).detach()
            # shape= batch_size, y_class_num, nt
            
            log_prob_y = F.log_softmax(y_logits, dim=1).unsqueeze(2).repeat(1, 1, self.d_nt)
            # shape= batch_size, y_class_num, nt
            
            stream_d_loss = -torch.logsumexp(log_prob_d + log_prob_y, dim=1)
            # shape= batch_size, nt
            
            stream_d_loss_mask = d_mask.float()
            stream_d_loss = stream_d_loss * stream_d_loss_mask / torch.sum(stream_d_loss_mask)
            stream_d_loss = stream_d_loss * self.weights.view((1, -1))
            # shape= batch_size, nt
            
            test_mask = 1 - y_mask.view((-1)).float()
            test_acc = (torch.argmax(y_logits, dim=1) == y.view((-1)).float()) * test_mask.float()
            test_acc = torch.sum(test_acc) / torch.sum(test_mask)
            
            stream_d_loss = torch.sum(stream_d_loss)
            
            stream_loss = stream_d_loss + y_reg_loss
            
            self.optimizer_xy.zero_grad()
            stream_loss.backward()
            self.optimizer_xy.step()
            return {
                "stream_d_loss": stream_d_loss.item(),
                "y_reg_loss": y_reg_loss.item(),
            }
            
    def cal_weights(self, y, d):
        assert self.y_class_num == 2
        self.conditional_entropy = []
        for i in range(self.d_nt):
            pdy = torch.zeros((self.d_size, 2))
            for vy in range(2):
                for vd in range(self.d_size):
                    pdy[vd, vy] = torch.mean(torch.logical_and(y == vy, d[:, i, 0] == vd).float()).item()
            d_onehot = F.one_hot(d[:, i, 0], self.d_size).float()
            pd = torch.mean(d_onehot, dim=0).view((-1, 1))
            ce_y_d = torch.nansum(pdy * torch.log(pd) - pdy * torch.log(pdy))
            self.conditional_entropy.append(ce_y_d)
        self.conditional_entropy = torch.stack(self.conditional_entropy)
        self.conditional_entropy /= torch.max(self.conditional_entropy)
        self.nt = torch.tensor(self.nt).float()
        self.nt = self.nt / torch.max(self.nt)
        nt_weights = torch.exp(-self.beta * self.nt)
        ce_weights = torch.exp(-self.alpha * self.conditional_entropy)
        ce_weights = ce_weights / torch.max(ce_weights)
        nt_correction = 1.0 / torch.tensor(list(range(self.d_nt, 0, -1))).float()
        self.weights = (nt_weights * ce_weights)
        self.weights /= torch.mean(self.weights)
        self.weights = (self.weights * nt_correction).to(self.device)
        return self.conditional_entropy

    def pretrain(self):
        pretrain_model_local_path = "./pretrain_model/criteo_pretrain.pt"
        if os.path.isfile(pretrain_model_local_path):
            print("loading pretrain model, p_y_x")
            self.p_y_x.load_state_dict(torch.load(pretrain_model_local_path))
            self.reg_p_y_x.load_state_dict(torch.load(pretrain_model_local_path))
            self.reg_p_y_x.eval()
            self.p_y_x.train()
        else:
            print("pretrain model not found, training")
            assert self.method == "pretrain", "pretrain model not found, but method is not pretrain"
            x, y, d, y_mask, d_mask, test_mask = self.pretrain_dataset["x"], self.pretrain_dataset["y"], self.pretrain_dataset["d"], \
                self.pretrain_dataset["y_mask"], self.pretrain_dataset["d_mask"], self.pretrain_dataset["test_mask"]
            x, y, d, y_mask, d_mask = get_train_data_from_batch(
                x, y, d, y_mask, d_mask, test_mask)
            for i_epoch in range(self.params["pretrain_epochs"]):
                print("pretrain epoch {}".format(i_epoch))
                self.update_batch(x, y, d, y_mask, d_mask, streaming=False)
            torch.save(self.p_y_x.state_dict(), pretrain_model_local_path)
            self.reg_p_y_x.load_state_dict(self.p_y_x.state_dict())
            self.reg_p_y_x.eval()
            return
            
        if self.method in ["pretrain", "ce", "oracle"]:
            return
        x, y, d, y_mask, d_mask, test_mask = self.pretrain_dataset["x"], self.pretrain_dataset["y"], self.pretrain_dataset["d"], \
            self.pretrain_dataset["y_mask"], self.pretrain_dataset["d_mask"], self.pretrain_dataset["test_mask"]
        x, y, d, y_mask, d_mask = get_train_data_from_batch(
            x, y, d, y_mask, d_mask, test_mask)
        if self.method == "gdfm":
            self.cal_weights(y, d)
        for i_epoch in range(self.params["pretrain_epochs"]):
            print("pretrain epoch {}".format(i_epoch))
            self.update_batch(x, y, d, y_mask, d_mask, streaming=False)

    def update_batch(self, x, y, d, y_mask, d_mask, streaming):
        if streaming:
            if self.method in ["pretrain"]:
                return
        perm = np.random.permutation(x.shape[0])
        x = x[perm]
        y = y[perm]
        d = d[perm]
        y_mask = y_mask[perm]
        d_mask = d_mask[perm]

        x_batchs = x.split(self.params["batch_size"])
        y_batchs = y.split(self.params["batch_size"])
        d_batchs = d.split(self.params["batch_size"])
        y_mask_batchs = y_mask.split(self.params["batch_size"])
        d_mask_batchs = d_mask.split(self.params["batch_size"])

        metrics = MetricAccumulator()
        if streaming:
            update_steps = self.params["update_steps"]
        else:
            update_steps = 1

        if self.method == "gdfm":
            self.p_d_xy.train()
        self.p_y_x.train()
        for i_epoch in range(update_steps):
            progress_info = ProgressInfo(total_step=len(x_batchs),
                                         prefix="update_batch",
                                         log_steps=self.params["log_steps"])

            for _x, _y, _d, _y_mask, _d_mask in zip(x_batchs, y_batchs, d_batchs, y_mask_batchs, d_mask_batchs):
                progress_info.step()
                _x = _x.to(self.device)
                _y = _y.to(self.device)
                _d = _d.to(self.device)
                _y_mask = _y_mask.to(self.device)
                _d_mask = _d_mask.to(self.device)
                loss_info = self.update_step(
                    _x, _y, _d, _y_mask, _d_mask, streaming)
                if (loss_info is not None) and (not streaming):
                    metrics.add(loss_info)
                if progress_info.on_log_steps() and (not streaming):
                    print(metrics)

                if self.debug:
                    break

    def predict(self, x):
        self.p_y_x.eval()
        probs = []
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        chunk_x = torch.split(x, self.params["test_batch_size"], dim=0)
        for _x in chunk_x:
            _x = _x.to(self.device)
            with torch.no_grad():
                if self.method != "gdfm":
                    logits = self.p_y_x(_x)
                    prob = F.softmax(logits, dim=1).detach().cpu()
                else:
                    y_logits = self.p_y_x(_x)
                    prob = F.softmax(y_logits, dim=1).detach().cpu()
            probs.append(prob)
        return {"prob": torch.cat(probs, dim=0)}

    def stream_train_and_predict(self):
        stream_metric = MetricAccumulator()
        stream_pred = []
        progress_info = ProgressInfo(total_step=len(
                                    self.stream_dataset), 
                                     prefix="stream", 
                                     log_steps=self.params["log_steps"]
                                     )
        for batch in self.stream_dataset:
            train_batch, test_batch = batch
            progress_info.step()
            test_x, test_y, test_d, test_y_mask, test_d_mask = test_batch

            test_batch_size = test_x.shape[0]
            if test_batch_size > 0:
                pred = self.predict(test_x)
                pred["test_y"] = test_y
                stream_pred.append(pred)
                batch_metric = cal_metric(pred, test_y)
                stream_metric.add(batch_metric, test_x.shape[0])

            if progress_info.on_log_steps():
                print(stream_metric)

            train_x, train_y, train_d, train_y_mask, train_d_mask = train_batch
            train_batch_size = train_x.shape[0]
            if train_batch_size > 0:
                self.update_batch(train_x, train_y, train_d,
                                  train_y_mask, train_d_mask, streaming=True)
            if self.debug:
                break

        return {
            "metric": stream_metric,
            "pred": stream_pred
        }
