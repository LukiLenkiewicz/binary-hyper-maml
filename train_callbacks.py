from collections import defaultdict
import os
import json
from pathlib import Path
from time import time

import lightning.pytorch as pl
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler


PRINT_FREEQ = 10

class FSLModule(pl.LightningModule):
    def __init__(self, model, params, neptune_run):
        super().__init__()
        # self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = model
        self.params = params
        self.neptune_run = neptune_run

        self.metrics = None
        self.delta_params_list = []

    def forward(self, x):
        print("#######################################################")
        print("############### FORWARD DOES SOMETHING ################")
        print("#######################################################")
        out = self.model.feature_query.forward(x)
        scores = self.model.classifier.forward(out)
        return scores

    def on_fit_start(self):
        self.max_acc = 0
        self.max_train_acc = 0
        self.max_acc_adaptation_dict = {}

        if self.params.hm_set_forward_with_adaptation:
            self.max_acc_adaptation_dict = {}
            for i in range(self.params.hn_val_epochs + 1):
                if i != 0:
                    self.max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = 0
                self.max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = 0

        if not os.path.isdir(self.params.checkpoint_dir):
            os.makedirs(self.params.checkpoint_dir)

        if (Path(self.params.checkpoint_dir) / "metrics.json").exists() and self.params.resume:
            with (Path(self.params.checkpoint_dir) / "metrics.json").open("r") as f:
                try:
                    self.metrics_per_epoch = defaultdict(list, json.load(f))
                    try:
                        self.max_acc = self.metrics_per_epoch["accuracy/val_max"][-1]
                        self.max_train_acc = self.metrics_per_epoch["accuracy/train_max"][-1]

                        if self.params.hm_set_forward_with_adaptation:
                            for i in range(self.params.hn_val_epochs + 1):
                                if i != 0:
                                    self.max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = \
                                    self.metrics_per_epoch[f"accuracy/val_support_max@-{i}"][-1]
                                self.max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = \
                                self.metrics_per_epoch[f"accuracy/val_max@-{i}"][-1]
                    except:
                        self.max_acc = self.metrics_per_epoch["accuracy_val_max"][-1]
                        self.max_train_acc = self.metrics_per_epoch["accuracy_train_max"][-1]
                except:
                    self.metrics_per_epoch = defaultdict(list)

        else:
            self.metrics_per_epoch = defaultdict(list)


    def on_train_epoch_start(self) -> None:
        self.avg_loss = 0
        self.task_count = 0
        self.loss_all = []
        self.acc_all = []
        self.optimizer.zero_grad()

        self.delta_list = []


    def training_step(self, batch, batch_idx):
        self.model.train()
        x, _ = batch
        optimizer = self.optimizers()

        self.n_query = x.size(1) - self.model.n_support
        assert self.n_way == x.size(0), "MAML do not support way change"

        loss, task_accuracy = self.model.set_forward_loss(x)
        self.avg_loss = self.avg_loss + loss.item()  # .data[0]
        self.loss_all.append(loss)
        self.acc_all.append(task_accuracy)

        task_count += 1

        if self.task_count == self.model.n_task:  # MAML update several tasks at one time
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()

            optimizer.step()

            task_count = 0
            loss_all = []


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % PRINT_FREEQ == 0:
            print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.model.epoch, self.model.stop_epoch, batch_idx,
                                                                            self.base_loader_len,
                                                                            self.avg_loss / float(batch_idx + 1)))

    def on_train_epoch_end(self):
        acc_all = np.asarray(self.acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean}

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        self.metrics = metrics
        scheduler = self.lr_schedulers()
        scheduler.step()

        delta_params = self.metrics.pop('delta_params', None)
        if delta_params is not None:
            self.delta_params_list.append(delta_params)

    def on_validation_epoch_start(self):
        self.acc_all = []
        self.delta_list = []

        self.eval_time = 0

    def validation_step(self, batch, batch_idx):
        x, _, = batch
        self.n_query = x.size(1) - self.model.n_support
        assert self.model.n_way == x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
        s = time()
        correct_this, count_this = self.model.correct(x)
        t = time()
        self.acc_all.append(correct_this / count_this * 100)
        self.eval_time += (t - s)

    def on_validation_epoch_end(self):
        iter_num = len(self.trainer.val_dataloaders)
        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k, v) in self.model.acc_at.items()
        }

        num_tasks = len(self.acc_all)
        acc_all = np.asarray(self.acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        ret.append(metrics)
        self.acc = ret

        acc, test_loop_metrics = self.acc

        print(f"Epoch {self.current_epoch}/{self.trainer.max_epochs}  | Max test acc {self.max_acc:.2f} | Test acc {acc:.2f} | Metrics: {test_loop_metrics}")

        metrics = metrics or dict()
        scheduler = self.lr_schedulers()
        metrics["lr"] = scheduler.get_lr()[0]
        metrics["accuracy/val"] = acc
        metrics["accuracy/val_max"] = max_acc
        metrics["accuracy/train_max"] = max_train_acc
        metrics = {
            **metrics,
            **test_loop_metrics,
            **self.max_acc_adaptation_dict
        }

        if self.params.hm_set_forward_with_adaptation:
            for i in range(self.params.hn_val_epochs + 1):
                if i != 0:
                    metrics[f"accuracy/val_support_max@-{i}"] = self.max_acc_adaptation_dict[
                        f"accuracy/val_support_max@-{i}"]
                metrics[f"accuracy/val_max@-{i}"] = self.max_acc_adaptation_dict[f"accuracy/val_max@-{i}"]

        if metrics["accuracy/train"] > max_train_acc:
            max_train_acc = metrics["accuracy/train"]

        if self.params.hm_set_forward_with_adaptation:
            for i in range(self.params.hn_val_epochs + 1):
                if i != 0 and metrics[f"accuracy/val_support_acc@-{i}"] > self.max_acc_adaptation_dict[
                    f"accuracy/val_support_max@-{i}"]:
                    self.max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"] = metrics[
                        f"accuracy/val_support_acc@-{i}"]

                if metrics[f"accuracy/val@-{i}"] > self.max_acc_adaptation_dict[f"accuracy/val_max@-{i}"]:
                    self.max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = metrics[f"accuracy/val@-{i}"]

        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...")
            max_acc = acc
            outfile = os.path.join(self.params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': self.current_epoch, 'state': self.model.state_dict()}, outfile)

            if self.params.maml_save_feature_network and self.params.method in ['maml', 'hyper_maml','bayes_hmaml']:
                outfile = os.path.join(self.params.checkpoint_dir, 'best_feature_net.tar')
                torch.save({'epoch': self.current_epoch, 'state': self.model.feature.state_dict()}, outfile)

        outfile = os.path.join(self.params.checkpoint_dir, 'last_model.tar')
        torch.save({'epoch': self.current_epoch, 'state': self.model.state_dict()}, outfile)

        if self.params.maml_save_feature_network and self.params.method in ['maml', 'hyper_maml','bayes_hmaml']:
            outfile = os.path.join(self.params.checkpoint_dir, 'last_feature_net.tar')
            torch.save({'epoch': self.current_epoch, 'state': self.model.feature.state_dict()}, outfile)

        if (self.current_epoch % self.params.save_freq == 0) or (self.current_epoch == self.trainer.max_epochs - 1):
            outfile = os.path.join(self.params.checkpoint_dir, '{:d}.tar'.format(self.current_epoch))
            torch.save({'epoch': self.current_epoch, 'state': self.model.state_dict()}, outfile)

        if metrics is not None:
            for k, v in metrics.items():
                self.metrics_per_epoch[k].append(v)

        with (Path(self.params.checkpoint_dir) / "metrics.json").open("w") as f:
            json.dump(self.metrics_per_epoch, f, indent=2)

        if self.neptune_run is not None:
            for m, v in metrics.items():
                self.neptune_run[m].append(v, step=self.current_epoch)

    def on_fit_end(self):
        if self.neptune_run is not None:
            self.neptune_run["best_model"].track_files(os.path.join(self.params.checkpoint_dir, 'best_model.tar'))
            self.neptune_run["last_model"].track_files(os.path.join(self.params.checkpoint_dir, 'last_model.tar'))

            if self.params.maml_save_feature_network:
                self.neptune_run["best_feature_net"].track_files(os.path.join(self.params.checkpoint_dir, 'best_feature_net.tar'))
                self.neptune_run["last_feature_net"].track_files(os.path.join(self.params.checkpoint_dir, 'last_feature_net.tar'))

        if len(self.delta_params_list) > 0 and self.params.hm_save_delta_params:
            with (Path(self.params.checkpoint_dir) / f"delta_params_list_{len(self.delta_params_list)}.json").open("w") as f:
                json.dump(self.delta_params_list, f, indent=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.params.milestones,
                                        gamma=0.3),
        return {"optimizer": optimizer, "scheduler": scheduler}