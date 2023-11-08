from collections import defaultdict
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch

PRINT_FREEQ = 10

class FSLModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, max_accuracy, max_acc_adaptation_dict, metrics_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_accuracy = max_accuracy
        self.max_acc_adaptation_dict = max_acc_adaptation_dict
        self.metrics_per_epoch = metrics_per_epoch

        self.metrics = None
        self.avg_loss = 0
        self.delta_params_list = []

        self.acc_all = []
        self.loss_all = []

    def forward(self, x):
        out = self.model.feature_query.forward(x)
        scores = self.model.classifier.forward(out)
        return scores

    def training_step(self, batch, batch_idx):
        x, _ = batch

        self.n_query = x.size(1) - self.model.n_support
        assert self.n_way == x.size(0), "MAML do not support way change"

        loss, task_accuracy = self.model.set_forward_loss(x)
        self.avg_loss = self.avg_loss + loss.item()  # .data[0]
        loss_all.append(loss)
        self.acc_all.append(task_accuracy)

        task_count += 1

        if task_count == self.model.n_task:  # MAML update several tasks at one time
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()

            self.optimizer.step()

            task_count = 0
            loss_all = []

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        self.n_query = x.size(1) - self.n_support
        assert self.n_way == x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
        s = time()
        correct_this, count_this = self.model.correct(x)
        t = time()
        self.acc_all.append(correct_this / count_this * 100)
        eval_time += (t - s)


class TrainCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        # beginningo of train func
        pass

    def on_fit_end(self, trainer, pl_module):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.avg_loss = 0
        pl_module.optimizer.zero_grad()

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx):
        i = batch_idx
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        if i % PRINT_FREEQ == 0:
            print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(pl_module.model.epoch, pl_module.model.stop_epoch, i,
                                                                            len(trainer.train_loader),
                                                                            pl_module.avg_loss / float(i + 1)))

        pl_module.metrics = {"accuracy/train": acc_mean}

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.model.acc_all = []
        self.delta_list = []
        acc_at = defaultdict(list)

        pl_module.model.acc_at = acc_at

        eval_time = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        iter_num = len(trainer.val_dataloaders)
        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k, v) in pl_module.model.acc_at.items()
        }

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        ret.append(metrics)
        pl_module.ret = ret