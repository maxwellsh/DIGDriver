import numpy as np
import torch
import scipy
import torch.utils.data
#from sklearn.metrics import r2_score4
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def r2_score(y_true, y_pred):
    r2 = scipy.stats.pearsonr(y_true, y_pred)[0]**2
    return r2 if not np.isnan(r2) else 0


class NNTrainer:
    def __init__(self, model, optimizer, loss_fn, bs, label_ids, train_ds, test_ds, device,  writer=None, get_attention_maps=False):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.bs = bs

        self.train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=16)
        self.test_dataloader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=16)

        self.label_ids = label_ids
        self.get_attention_maps = get_attention_maps

        self.writer = writer

        '''
        if writer is not None:
            shape = train_ds.get_data_shape()
            dummy_input = (torch.zeros(1, shape[1], shape[2]),)
            print(dummy_input[0].size())
            self.writer.add_graph(model(), dummy_input, True)
        '''

    def train(self, epoch, run, print_interval=10, autoreg=False):
        # toggle model to train mode
        self.model.train()

        samp_ctr = 0
        batch_num = len(self.train_dataloader)
        loss_sums = np.zeros(len(self.label_ids))
        corr_coef_sums = np.zeros(len(self.label_ids))
        all_preds = [[] for _ in range(len(self.label_ids))]
        all_true = [[] for _ in range(len(self.label_ids))]
        all_features_lst = [[] for _ in range(len(self.label_ids))]
        print('Training epoch {}'.format(epoch))
        for j, batch in enumerate(self.train_dataloader):
            t_lst = batch[-1]
            if autoreg == True:
                y_lst, fv_lst,  _ = self.model(batch[0].to(self.device), torch.cat(batch[1], dim=1).to(self.device))
            else:
                y_lst, fv_lst,  _ = self.model(batch[0].to(self.device))
            samp_ctr += batch[0].size()[0]
            loss_lst = []
            for i, t in enumerate(t_lst):
                y = y_lst[i]
                all_preds[i].extend(y.data.cpu().numpy().tolist())
                all_true[i].extend(t.data.cpu().numpy().tolist())
                all_features_lst[i].extend(fv_lst[i].data.cpu().numpy())
                task_loss = self.loss_fn(y, t.to(self.device))# + torch.norm(attention, p=1, dim=(1,2)).mean()
                loss_lst.append(task_loss)
                loss_sums[i] += task_loss.item()
                corr_coef = r2_score(t.data.cpu().numpy(), y.data.cpu().numpy())
                corr_coef_sums[i] += corr_coef

            loss = torch.sum(torch.stack(loss_lst))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if j % int(batch_num * print_interval / 100) == 0 and j > 0:  # print progress every print_interval%
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}\tAccuracy: {}'.format(
                    epoch, j, batch_num, 100. * j / batch_num,
                    loss_sums / (samp_ctr / self.bs), corr_coef_sums / (samp_ctr / self.bs)))

        train_accs = corr_coef_sums / batch_num
        train_losses = loss_sums / batch_num

        print('====> Epoch: {}, Average loss: {}, Average accuracy: {}'.format(epoch, train_losses, train_accs))

        if self.writer is not None:
            for i in range(len(self.label_ids)):
                self.writer.add_scalar('Train_{}/Loss_{}'.format(run, self.label_ids[i]), train_losses[i], epoch)
                self.writer.add_scalar('Train_{}/R^2_{}'.format(run, self.label_ids[i]), train_accs[i], epoch)

        return train_losses, train_accs, all_features_lst, all_preds, all_true,

    def predict(self, model, dataloader, epoch, run, set_id='Test', autoreg=False):
        # toggle model to test / inference mode
        model.eval()

        batch_num = len(dataloader)
        loss_sums = np.zeros(len(self.label_ids))
        corr_coef_sums = np.zeros(len(self.label_ids))
        all_preds = [[] for _ in range(len(self.label_ids))]
        all_true = [[] for _ in range(len(self.label_ids))]
        all_features_lst = [[] for _ in range(len(self.label_ids))]
        all_att = []
        for j, batch in enumerate(dataloader):
            t_lst = batch[-1]
            if autoreg == True:
                y_lst, fv_lst, attention = self.model(batch[0].to(self.device), torch.cat(batch[1], dim=1).to(self.device))
            else:
                y_lst, fv_lst, attention = self.model(batch[0].to(self.device))

            if self.get_attention_maps: all_att.append(attention.cpu().detach().numpy())
            with torch.no_grad():
                for i, t in enumerate(t_lst):
                    y = y_lst[i]
                    all_features_lst[i].append(fv_lst[i].cpu().detach().numpy())
                    all_preds[i].extend(y.data.cpu().numpy().tolist())
                    all_true[i].extend(t.data.cpu().numpy().tolist())
                    corr_coef_sums[i] += r2_score(t.data.cpu().numpy(), y.data.cpu().numpy())
                    loss_sums[i] += self.loss_fn(y, t.to(self.device))# + torch.norm(attention, p=1, dim=(1,2)).mean()
        all_features = [np.concatenate(all_features_lst[j], axis=0) for j in range(len(all_features_lst))]
        test_accs = corr_coef_sums / batch_num
        test_losses = loss_sums / batch_num

        print('====> Test set loss: {}, accuracy: {}'.format(test_losses, test_accs))

        if self.writer is not None:
            for i in range(len(self.label_ids)):
                self.writer.add_scalar('{}_{}/Loss_{}'.format(set_id, run, self.label_ids[i]), test_losses[i], epoch)
                self.writer.add_scalar('{}_{}/R^2_{}'.format(set_id, run, self.label_ids[i]), test_accs[i], epoch)

            for name, param in self.model.named_parameters():
                if 'bn' not in name:
                    self.writer.add_histogram(name, param, epoch)

            self.plot_prediction_scatter(dataloader, all_preds, '{}/run_{}/epoch_{}'.format(set_id, run, epoch), test_accs)
            self.plot_prediction_histogram(dataloader, all_preds, '{}/run_{}/epoch_{}'.format(set_id, run, epoch), test_accs)

        if self.get_attention_maps:
            return test_losses, test_accs, all_features, all_preds, all_true, np.concatenate(all_att, axis=0)
        else:
            return test_losses, test_accs, all_features, all_preds, all_true, None

    def test(self, epoch, run, autoreg=False):
        return self.predict(self.model, self.test_dataloader, epoch, run, autoreg=autoreg)

    def plot_prediction_scatter(self, dataloader, preds, writer_id, accs, confidence=99.9):
        for i in range(len(preds)):
            t = np.concatenate([l[i].data.cpu().numpy() for (_,l) in dataloader])
            fig = plt.figure()
            ax = plt.gca()
            y = np.array(preds[i])
            ax.scatter(t, y, alpha=0.3)
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x)
            ax.set_ylim(0, np.percentile(y, confidence) + 1)
            ax.set_xlim(0, np.percentile(t, confidence) + 1)
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title('Accuracy: {}'.format(np.round(accs[i], 3)))
            self.writer.add_figure('{}/{}/Scatter'
                                   .format(self.label_ids[i], writer_id), fig)

    def plot_prediction_histogram(self, dataloader, preds, writer_id, accs, confidence=99.85):
        for i in range(len(preds)):
            t = np.concatenate([l[i].data.cpu().numpy() for (_,l) in dataloader])
            fig = plt.figure()
            ax = plt.gca()
            y = np.array(preds[i])
            y_max_bin = int(np.percentile(y, confidence) + 1)
            t_max_bin = int(np.percentile(t, confidence) + 1)
            max_bin = max(y_max_bin, t_max_bin)
            ax.hist(t, max_bin, (0, max_bin), alpha=0.5)
            ax.hist(y, max_bin, (0, max_bin), alpha=0.5)
            ax.set_xlabel('Mutation Count')
            ax.set_ylabel('Window #')
            ax.set_title('Accuracy: {}'.format(np.round(accs[i], 3)))
            ax.legend(['True', 'Predicted'])
            self.writer.add_figure('{}/{}/Histogram'
                                   .format(self.label_ids[i], writer_id), fig)

