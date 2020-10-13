import copy
import logging
from collections import defaultdict
from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .xref_model import XrefModel


def reduce_mean(data):
    return data.mean(axis=0)


def reduce_max_confidence(data):
    best_idx = np.abs(data[:, 0] - data[:, 1]).argmax()
    return data[best_idx]


class XrefTorchModel(XrefModel):
    def __init_subclass__(cls):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def device(self):
        if getattr(self, "__device", None) is None:
            self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.__device

    def dumps(self):
        self.clf["embedding"] = self.clf["embedding"].cpu()
        result = super().dumps()
        self.clf["embedding"] = self.clf["embedding"].to(self.device)
        return result

    @classmethod
    def sample_weighted_criterion(cls, criterion):
        def _(predict, truth, weight):
            loss = criterion(predict, truth)
            loss *= weight
            return loss.sum() / predict.shape[0]

        return _

    def create_dataloader_samples(
        self, samples, shuffle=True, filtered_idxs=None, sample_group_len=None
    ):
        data_loader = DataLoader(
            samples,
            batch_size=self.meta["batch_size"],
            shuffle=shuffle,
            collate_fn=self.transform_data,
            pin_memory=True,
            drop_last=False,
            num_workers=6,
        )
        data_loader.filtered_idxs = filtered_idxs or []
        data_loader.sample_group_len = sample_group_len or []
        return data_loader

    def _train(self, data, model, optimizer, criterion):
        train_loss = 0
        train_acc = 0
        N = 0
        model = model.train()
        for i, X in enumerate(
            tqdm(data, desc="Training", leave=True, unit_scale=self.meta["batch_size"])
        ):
            N += X["judgement"].shape[0]
            X = {k: v.to(self.device) for k, v in X.items()}
            optimizer.zero_grad()
            output = model(**X)
            loss = criterion(output, X["judgement"], X["weight"])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == X["judgement"]).sum().item()

        return train_loss / N, train_acc / N

    def _test(self, data, model, criterion):
        loss = 0
        acc = 0
        N = 0
        N_positive = 0
        outputs = []
        model = model.eval()
        for X in tqdm(
            data, desc="Testing", leave=True, unit_scale=self.meta["batch_size"]
        ):
            with torch.no_grad():
                N += X["judgement"].shape[0]
                X = {k: v.to(self.device) for k, v in X.items()}
                output = model(**X)
                N_positive += output[:, -1].sum().item()
                outputs.append(output.cpu())
                loss = criterion(output, X["judgement"], X["weight"])
                loss += loss.item()
                acc += (output.argmax(1) == X["judgement"]).sum().item()
        logging.debug(f"n_positive_pct: {N_positive / N}")
        return loss / N, acc / N

    def fit_torch(
        self,
        model,
        train_data,
        test_data,
        criterion,
        optimizer,
        scheduler,
        max_patience=15,
    ):
        best_model = None
        best_loss = None
        cur_patience = max_patience
        history = defaultdict(list)
        for epoch in count(0):
            train_loss, train_acc = self._train(train_data, model, optimizer, criterion)
            scheduler.step()
            valid_loss, valid_acc = self._test(test_data, model, criterion)

            logging.debug(f"Epoch: {epoch + 1}")
            logging.debug(
                f"\tLoss: {train_loss:.4e}(train)\t|\tAcc: {train_acc * 100:.3f}%(train)"
            )
            logging.debug(
                f"\tLoss: {valid_loss:.4e}(valid)\t|\tAcc: {valid_acc * 100:.3f}%(valid)"
            )

            history["valid_loss"].append(valid_loss)
            history["valid_acc"].append(valid_acc)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if best_loss is None or valid_loss < best_loss:
                logging.debug(f"New best model: {valid_loss} < {best_loss}")
                best_loss = valid_loss
                best_model = copy.copy(model.state_dict())
                cur_patience = max_patience
            else:
                cur_patience -= 1
                logging.debug(f"Current Patience: {cur_patience}")
            if cur_patience == 0:
                logging.info("Out of patience.")
                break
        model.load_state_dict(best_model)
        return model, history

    def merge_sample_groups(self, sample_group_len, output, reduction_fxn):
        n_groups = len(sample_group_len)
        N = output.shape[0]
        M = N - sum(length - 1 for idx, length in sample_group_len)
        reduced_output = np.zeros((M, 2))
        group_idx = 0
        o_idx = 0
        r_idx = 0
        while r_idx < reduced_output.shape[0]:
            if group_idx < n_groups and o_idx == sample_group_len[group_idx][0]:
                group_length = sample_group_len[group_idx][1]
                group_data = output[o_idx : o_idx + group_length]
                reduced_output[r_idx] = reduction_fxn(group_data)
                o_idx += group_length
                group_idx += 1
            else:
                reduced_output[r_idx] = output[o_idx]
                o_idx += 1
            r_idx += 1

        return reduced_output

    def unpack_output(self, dataloader, output, reduction=None):
        sample_group_len = dataloader.sample_group_len
        if sample_group_len:
            logging.debug("Unpacking grouped results")
            if reduction is None or reduction == "max_confidence":
                reduction_fxn = reduce_max_confidence
            elif reduction == "mean":
                reduction_fxn = reduce_mean
            else:
                raise ValueError(f"Unknown reduction function: {reduction}")
            output = self.merge_sample_groups(sample_group_len, output, reduction_fxn)

        skipped_idxs = dataloader.filtered_idxs
        if skipped_idxs:
            logging.debug("Filling filtered results")
            skipped_idxs.sort()
            idxs = [si - i for i, si in enumerate(skipped_idxs)]
            output = np.insert(output, idxs, [0.5, 0.5], axis=0)
        return output

    def predict_torch(self, model, dataloader, reduction=None):
        model = model.to(self.device).eval()
        N = len(dataloader.dataset)
        result = np.zeros((N, 2))
        with torch.no_grad():
            base_i = 0
            for X in dataloader:
                X = {k: v.to(self.device) for k, v in X.items()}
                output = model(**X).cpu()
                n = output.shape[0]
                result[base_i : base_i + n] = output
                base_i += n
        return self.unpack_output(dataloader, result, reduction=reduction)
