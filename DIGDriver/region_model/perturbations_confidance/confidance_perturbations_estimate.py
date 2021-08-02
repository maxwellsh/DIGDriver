import os
import sys
import json
import copy
import h5py
import numpy as np
import torch
from types import SimpleNamespace
from torch import nn
from sklearn.metrics import r2_score

#from nets.nets import *
#from nets.trainer import *

def add_noise_to_model(model, noise):
    tmp_model = copy.deepcopy(model).cuda()
    with torch.no_grad():
        for param in tmp_model.parameters():
            print
            param.add_(torch.normal(0, noise, param.size()).cuda())
    return tmp_model


def compute_confidance(preds, labels):
    confs = np.empty((preds.shape[0], preds.shape[2]))
    means = np.empty((preds.shape[0], preds.shape[2]))
    accs = np.empty(preds.shape[0])
    for i in range(preds.shape[0]):
        for j in range(preds.shape[2]):
            confs[i, j] = np.std(preds[i, :, j])
            means[i, j] = np.mean(preds[i, :, j])
        accs[i] = r2_score(means[i], labels)
    return means, confs, accs


def test_confidance(model, data, labels, loss_fn, params, verbose=False):
     # toggle model to test / inference mode
    model.eval()
        
    # round sample num to full batches
    samp_num = len(labels) - len(labels) % params.bs

    preds = np.empty((len(params.alphas), params.reps, samp_num))
    for i, alpha in enumerate(params.alphas):
        for rep in range(params.reps):
            loss_sum = 0
            acc_sum = 0 
            tmp_model = add_noise_to_model(model, alpha)
            for b_samp in range(0, samp_num, params.bs):
                x = torch.tensor(data[b_samp:b_samp + params.bs]).float().cuda()
                with torch.no_grad():
                    y = tmp_model(x)
                t = torch.tensor(labels[b_samp:b_samp + params.bs]).float().cuda()

                loss_sum += loss_fn(y, t).item()
                acc_sum += r2_score(t.data.cpu().numpy(), y.data.cpu().numpy())
                preds[i, rep, b_samp:b_samp + params.bs] = y.data.cpu().numpy()      

            if verbose:
                print('Repetition {} alpha: {}, loss: {:.4f}, accuracy: {:.4f}'.format(rep, alpha, loss_sum / (samp_num / params.bs), acc_sum / (samp_num / params.bs)))

        print('Accuracy for alpha: {} over {} repetitions is: {}'.format(alpha, params.reps, r2_score(np.mean(preds[i], axis=0), labels[:samp_num])))
    
    return compute_confidance(preds, labels[:samp_num])


def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) < 2:
        config_file = os.path.join(cur_dir, "configs/config_confidance.json")
        print('No input was given, using {} as configuration file.'.format(config_file))
    else:
        config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    params = SimpleNamespace()
    params.reps = config['repetitions']
    params.alphas = config['alphas']
    params.bs = config['bs']
    
        
    data_file = os.path.join('models', 'test_data_' + config['model_file'] + '.h5')
    print('Loading data and labels from file {}...'.format(data_file))
    h5f = h5py.File(data_file, 'r')
    labels = h5f['labels'][:]
    data = h5f['data'][:]
             
    print('Loading model...')
    model = torch.load(os.path.join('models', 'best_model_' + config['model_file'] + '.pt')).cuda()
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for name, param in model.named_parameters():
            print(name, np.mean(param.detach().cpu().numpy()), np.std(param.detach().cpu().numpy()))

    print('Computing prediction and confidance...')
    preds, confidance, accs = test_confidance(model, data, labels, loss_fn, params)

    #TODO: add downstream task logic

    print('Done!')
    
if __name__ == '__main__':
    main()
