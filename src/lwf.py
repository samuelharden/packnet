"""Implements Learning without Forgetting (LwF)."""
from __future__ import division, print_function

import argparse
import copy
import json
import warnings
import os
import dataset
import networks as net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import utils as utils
from prune import SparsePruner
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from fastai.text import *
from fastai.core import *
from fastai.lm_rnn import *
import fastai.metrics as metrics
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode',
                   choices=['finetune', 'eval'],
                   help='Run mode')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune')
FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
FLAGS.add_argument('--lr', type=float,
                   help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--dropout', type=float, default=0.5,
                   help='Dropout ratio')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight decay')
FLAGS.add_argument('--temperature', type=float, default=2.0,
                   help='LwF logit temperature')
FLAGS.add_argument('--ft_shared_after', type=int, default=5,
                   help='Finetune shared layers after this epoch')
# Paths.
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * scale

class Stepper():
    def __init__(self, m, opt, crit, original_model, this_model, examples, example_labels, 
                 clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m.model,opt,crit,clip,reg_fn
        self.om = original_model
        self.tsne = TSNE(n_components=2, perplexity=30.0) 
        self.this_model = this_model
        self.modell = m
        self.examples = examples
        self.example_lbl = example_labels
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def transform_output(self, output):
      rout, osss = output
      X_view = osss[-1]
      sl,bs,_ = X_view.size()
      X_max = F.adaptive_max_pool1d(X_view.permute(1,2,0), (1,)).view(bs,-1)
      X_avg = F.adaptive_avg_pool1d(X_view.permute(1,2,0), (1,)).view(bs,-1)
      X_new = torch.cat([X_view[-1], X_max, X_avg], 1)
      return to_np(X_new)

    def distance(self, d1, d2):
      #print(d1, d2)
      return ((d1[0] - d2[0])**2 + (d1[1] - d1[1])**2)**(0.5)

    def compute_tsne_difference(self, tsne_data, num_examples):
      data = TSNE(n_components=2, perplexity=30.0).fit_transform(tsne_data)
      distance_loss = 0
      #print("TSNE DATA", data)
      for i in range(64):
        distance_loss += self.distance(data[i], data[i + num_examples])
      return distance_loss/64

    def step(self, xs, y, epoch):
        xtra = []
        ll = xs[0]
        #print(type(xs), xs)
        #print(type(ll), ll)
        #print("ll", ll.size())
        #print(self.examples, self.example_lbl)
        sys.stdout.flush()
        batch_original = ll.data.clone()
        batch_original = Variable(batch_original, requires_grad=False)
        batch_original = batch_original.cuda(0)
        self.om.cuda(0)
        orig_output = self.om.shared(batch_original)

        '''
        loss_tsne = 0
        ## CHange output to something which can be feed into tsne model
        tsne_data = []
        orig_e = Variable(self.examples,  requires_grad=False)
        this_e = Variable(self.examples,  requires_grad=False)
        orig_e.cuda(0)
        this_e.cuda(0)
        orig_examples = self.transform_output(self.om.shared(orig_e))
        this_examples = self.transform_output(self.this_model.shared(this_e))
        tsne_data.extend(orig_examples)
        tsne_data.extend(this_examples)
        loss_tsne = self.compute_tsne_difference(tsne_data, 64)
        #print("tsne loss", loss_tsne)
        '''
        #print(batch_original, orig_output)
        target_logits = [classifier(orig_output)[0].data.cpu()
                         for classifier in self.om.classifiers]
        # Move to same GPU as current model.
        target_logits = [Variable(item.cuda(), requires_grad=False)
                         for item in target_logits]
        scale = [item.size(-1) for item in target_logits]
        output = self.this_model.shared(*xs)
        pred_logits = [classifier(output) for classifier in self.this_model.classifiers]
        output = self.this_model.classifier(output)
        #output = self.m(*xs)
        # Compute loss.
        dist_loss = 0
        # Apply distillation loss to all old tasks.
        for idx in range(len(target_logits)):
            dist_loss += distillation_loss(
                pred_logits[idx][0], target_logits[idx], 2.0, scale[idx])
        #print("Dist loss", dist_loss, len(self.this_model.classifiers))
        if isinstance(output,tuple): output,*xtra = output
        if self.fp16: self.m.zero_grad()
        else: self.opt.zero_grad() 
        loss = raw_loss = self.crit(output, y) + dist_loss
#        raw_loss = loss + dist_loss + loss_tsne
        #print("Before loss", self.reg_fn, loss)
        if self.loss_scale != 1: assert(self.fp16); loss = loss*self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        #print("After loss", loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:   # Gradient clipping
            if IS_TORCH_04: nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else: nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0: 
            #Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        if self.fp16: 
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds,tuple): preds=preds[0]
        return preds, self.crit(preds, y)


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, original_model, model, dataset2idx):
        self.args = args
        self.cuda = args.cuda
        self.dataset2idx = dataset2idx
        self.original_model = original_model
        self.model = model

        if args.mode != 'check':
            # Set up data loader, criterion, and pruner.
            if 'cropped' in args.train_path:
                train_loader = dataset.train_loader_cropped
                test_loader = dataset.test_loader_cropped
            else:
                train_loader = dataset.train_loader
                test_loader = dataset.test_loader
            self.train_data_loader = train_loader(
                args.train_path, args.batch_size, pin_memory=args.cuda)
            self.test_data_loader = test_loader(
                args.test_path, args.batch_size, pin_memory=args.cuda)
            self.criterion = nn.CrossEntropyLoss()

    def get_examplers(self):
      num_examplers = 64
      num_datasets = len(self.dataset2idx)
      examples = []
      lbl = []
      if num_datasets < 2:
        return examples,lbl 
      num_datasets_examplers = num_examplers/float(num_datasets - 1)
      print("Number of datasets we have are: ", num_datasets, num_datasets_examplers)
      for d in self.dataset2idx:
        print("We have following datasets", d, self.dataset2idx[d], self.args.dataset)
        if d != self.args.dataset:
          print("Lets add this datasets to examplers")
          data_loader = dataset.test_loader(get_test_dataset(d), num_datasets_examplers, pin_memory=self.args.cuda)
          for btch,label in data_loader:
            #btch = data_loader.get_batch([0])
            examples = btch
            lbl = label
            break
      return examples,lbl
      
    def eval_all(self):
        for d in self.dataset2idx:
          self.model.set_dataset(d)
          print("Evaluating", d, self.dataset2idx[d])
          data_loader = dataset.test_loader(get_test_dataset(d), self.args.batch_size, pin_memory=self.args.cuda)
          self.eval_n(self.model, -1, data_loader, "unknown", d)
        self.model.set_dataset(self.args.dataset)

    def eval_all_layers(self):
      for layer in [0,1,2]:
        labels_all = []
        predictions_all = []
        tsne_all = []
        offset = 0
        for d in self.dataset2idx:
          print(d, self.dataset2idx[d], layer)
          data_loader = dataset.test_loader(get_test_dataset(d), self.args.batch_size, pin_memory=self.args.cuda)
          _,tsne,labels,predictions = self.eval_n(self.model, layer, data_loader, "unknown", d)
          tsne_all.extend(tsne)
          for l in range(len(labels)):
            labels_all.append(labels[l]+offset)
          predictions_all.extend(predictions)
          offset = offset + 2
        # data_view_tsne, labels_orig, predictions_orig, epoch, getlayer, dataset
        #self.run_tsne_embeddings(tsne_all, labels_all, predictions_all, "unknown", layer, "all_label_scaled")
 

    def eval(self):
      return self.eval_n(self.model, -1, self.test_data_loader, self.args.dataset, "normal")

    def eval_n(self, runmodel, getlayer, use_test_data_loader, epoch, dataset):
        """Performs evaluation."""
        print("Running evaluation")
        runmodel.eval()
        error_meter = None
        labels_orig = []
        predictions_orig = []
        tnse_embedding = []
        data_view_tsne = []
        print('Performing eval...')
        tsne_obj = TSNE(n_components=2, perplexity=30.0)
        for batch, label in tqdm(use_test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)
            #print("Batch",batch)
            raw_output = runmodel.shared(batch)
            tmp = copy.deepcopy(raw_output)
            rout, osss = tmp
            #print("Length of outputs", len(rout), len(osss), osss[0].shape, osss[1].shape, osss[2].shape)
            X_view = osss[getlayer]
            raw_output = self.model.classifier(raw_output)
            #print("Before", X_view.shape)
            #X_view = X_view.view(X_view.size(1), -1)
            sl,bs,_ = X_view.size()
            X_max = F.adaptive_max_pool1d(X_view.permute(1,2,0), (1,)).view(bs,-1)
            X_avg = F.adaptive_avg_pool1d(X_view.permute(1,2,0), (1,)).view(bs,-1)
            X_new = torch.cat([X_view[-1], X_max, X_avg], 1)
            #print("After SHape", X_new.shape)
            sys.stdout.flush()
            #print(X_embedded)
            output = raw_output[0]
            #print("OUtputs", output.size(), "Batch", batch.size())
            output_np = to_np(output)
            #print("Outputs", output_np)
            predictions = np.argmax(output_np, axis=1)
            #print("Predictions", predictions)
            label_np = to_np(label)
            #print("Labels", label_np)

            labels_orig.extend(label_np)
            predictions_orig.extend(predictions)
            #for i in range(len(X_view)):
            #  print(to_np(X_view[i])[:9000])
            #  data_view_tsne.append(to_np(X_view[i]))
            data_view_tsne.extend(to_np(X_new))
            #print("Before", len(data_view_tsne))

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)
        print(confusion_matrix(labels_orig, predictions_orig))
        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
        #self.run_tsne_embeddings(data_view_tsne, labels_orig, predictions_orig, epoch, getlayer, dataset)
        sys.stdout.flush() 
        self.model.train()
        return errors,data_view_tsne,labels_orig, predictions_orig


    def run_tsne_embeddings(self, data_view_tsne, labels_orig, predictions_orig, epoch, getlayer, dataset):
        tnse_embedding = TSNE(n_components=2, perplexity=30.0).fit_transform(data_view_tsne)
        #tnse_embedding = PCA(n_components=2).fit_transform(data_view_tsne)
        for i in range(0, len(tnse_embedding)):
          print(i,tnse_embedding[i],labels_orig[i], predictions_orig[i])
          if labels_orig[i] == 1:
            plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"ro")
          elif labels_orig[i] == 2:
            plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"go")
          elif labels_orig[i] == 3:
            plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"yo")
          else:
            plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"bo")
        plt.savefig("embedding_plot_epoch"+ str(epoch) + "layer_" + str(getlayer)+ "_dataset"+ dataset + ".png")
        #print("TSNSE array", tnse_embedding)
        #print("Labels, ", labels_orig, "predictions : ", predictions_orig)

    def do_batch(self, optimizer, batch, label, epoch_idx):
        """Runs model for one batch."""
        batch_original = batch.clone()
        if self.cuda:
            batch_original = batch_original.cuda(0)
            batch = batch.cuda()
            label = label.cuda()
        batch_original = Variable(batch_original, requires_grad=False)
        batch = Variable(batch)
        label = Variable(label)

        # Get targets using original model.
        #self.original_model.eval()
        #x = self.original_model.shared(batch_original)
        #target_logits = [classifier(x).data.cpu()
        #                 for classifier in self.original_model.classifiers]
        # Move to same GPU as current model.
        #target_logits = [Variable(item.cuda(), requires_grad=False)
        #                 for item in target_logits]
        #scale = [item.size(-1) for item in target_logits]

        # Work with current model.
        # Set grads to 0.

        # Do forward.
        optimizer.zero_grad()
        x = self.model.model(batch)
        #pred_logits = [classifier(x) for classifier in self.model.classifiers]

        # Compute loss.
        #dist_loss = 0
        # Apply distillation loss to all old tasks.
        #for idx in range(len(target_logits)):
        #    dist_loss += distillation_loss(
        #        pred_logits[idx], target_logits[idx], self.args.temperature, scale[idx])
        # Apply cross entropy for current task.
        #output = pred_logits[-1][0]
        #output = self.model.classifier(x)[0]
        #output = x[0]
        #output_np = to_np(output)
        #print("Outputs", output_np)
        #predictions = np.argmax(output_np, axis=1)
        #print("Output", output)
        #print("labels there", label)
        new_loss = self.criterion(x[0], label)
        loss =  new_loss
        print("Losss", loss.data)

        # Do backward.
        loss.backward()

#        if epoch_idx <= self.args.ft_shared_after:
            # Set shared layer gradients to 0 if early epochs.
#            for module in self.model.shared.modules():
#                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
#                    module.weight.grad.data.fill_(0)
#                    module.bias.grad.data.fill_(0)
            # Set old classifier gradients to 0 if early epochs.
#            for idx in range(len(self.model.classifiers) - 1):
#                module = self.model.classifiers[idx]
#                module.weight.grad.data.fill_(0)
#                module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        print("Epoch ", epoch_idx)
        #self.learn.fit(self.lrs, 1, wds=self.wd)
        callbacks = []
        batch_per_epoch = len(self.md.trn_dl)
        cl = 1
        self.wd_sched = WeightDecaySchedule(self.loptimizer, batch_per_epoch, cl, 1, 1,
                                                False, None)
        callbacks += [self.wd_sched]
        #s = Stepper(self.modell.model, self.loptimizer.opt, self.criterion)
        exampls,lbl = self.get_examplers()
        fit(self.modell, self.md, 1, self.loptimizer.opt, self.criterion,metrics=self.learn.metrics, stepper=Stepper,
         clip=self.learn.clip, reg_fn=self.learn.reg_fn, callbacks=callbacks, original_model=self.original_model, this_model=self.model,
         examples=exampls, example_labels=lbl)
         
        #for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
        #    self.do_batch(optimizer, batch, label, epoch_idx)
#        print("Params for shared")
#        for para in self.model.shared.parameters():
#          print("Is is grad true", para.requires_grad, type(para.data), para.size(), para.data)
#        print("Params for classifier")
#        for para in self.model.classifier.parameters():
#          print("Is is grad true", para.requires_grad, type(para.data), para.size(), para.data)
#        print("Params for model")
#        for para in self.model.model.parameters():
#          print("Is is grad true", para.requires_grad, type(para.data), para.size(), para.data)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module

        # Prepare the ckpt.
        ckpt = {
            'epoch': epoch,
            'dataset2idx': self.dataset2idx,
            'accuracy': best_accuracy,
            'errors': errors,
            'args': self.args,
            'model': base_model,
        }

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def train(self, epochs, optimizer, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        self.md = ModelData(self.args.train_path, self.train_data_loader, self.test_data_loader)
        bptt,em_sz,nh,nl = 70,400,1150,3
        opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
        dps = np.array([0.4,0.5,0.05,0.3,0.4])*1.0
        self.modell = TextModel(to_gpu(self.model.model))
        self.learn = RNN_Learner(self.md, TextModel(to_gpu(self.model.model)), opt_fn=opt_fn)
        self.learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        self.learn.clip=25.
        self.learn.metrics = [metrics.accuracy]
        lr=0.01
        lrm=2.6
        self.lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
        self.wd=1e-6
        self.learn.unfreeze()
        #self.learn.fit(self.lrs, 3, wds=self.wd)
        self.model.shared = self.model.model[0]
        self.model.classifier = self.model.model[1]
        optimizer = self.learn.get_layer_opt(self.lrs, self.wd).opt
        self.loptimizer = self.learn.get_layer_opt(self.lrs, self.wd)

        set_trainable(children(self.model.shared), True)
        set_trainable(children(self.model.classifier), True)

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            optimizer = utils.step_lr(epoch_idx, self.args.lr, self.args.lr_decay_every,
                                      self.args.lr_decay_factor, optimizer)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer)
            errors,_,_,_ = self.eval()
            self.eval_all()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))

def get_test_dataset(dataset):
  return '../data/%s/' % (dataset)

def get_train_dataset(dataset):
  return '../data/%s/' % (dataset)


def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    if not args.train_path:
        args.train_path = get_train_dataset(args.dataset)
    if not args.test_path:
        args.test_path = get_test_dataset(args.dataset)

    itos = pickle.load(open(args.train_path  + '/tmp/itos.pkl', 'rb'))
    vs = len(itos)

    exists = False
    if args.loadname:
      exists = os.path.isfile(args.loadname)
    print("Load model file available ? ", exists, args.loadname)
    # Load the required model.
    print("This", (not exists))
    if ('finetune' in args.mode) and (not exists):
        print("Loading new mode")
        model = net.TextModelMY()
        load_model(model.shared, args.train_path + "/models/fwd_pretrain_wiki_finetunelm_lm_enc.h5")
        dataset2idx = {}
    else:
        ckpt = torch.load(args.loadname)
        model = ckpt['model']
        dataset2idx = ckpt['dataset2idx']
        print("Loading the old saved model")

    original_model = copy.deepcopy(model)

    
    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.dataset not in dataset2idx:
        idxs = [dataset2idx[key] for key in dataset2idx]
        if len(idxs) == 0:
          dataset2idx[args.dataset] =  1
        else:
          dataset2idx[args.dataset] = max(idxs) + 1
    if args.cuda:
        model = model.cuda(0)
        if args.mode == 'finetune':
            original_model = original_model.cuda(0)

    # Create the manager object.
    manager = Manager(args, original_model, model, dataset2idx)

#    print("Running eval on old model")
#    print(manager.eval())

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        print("Evaluating all models before starting")
        manager.eval_all()
        print("Doing the fine tuning")
        model.set_dataset(args.dataset)
        # Get optimizer with correct params.
        if args.finetune_layers == 'all':
            print("Doing the fine tuning for all later")
            for idx, m in enumerate(model.modules()):
              print(idx, '->', m)
            params_to_optimize = [ {'params': model.shared.parameters()}, {'params': model.classifier.parameters()}]
        elif args.finetune_layers == 'classifier':
            for param in model.shared.parameters():
                param.requires_grad = False
            params_to_optimize = model.classifier.parameters()
        elif args.finetune_layers == 'fc':
            params_to_optimize = []
            # Add fc params.
            for param in model.shared.parameters():
                if param.size(0) == 4096:
                    param.requires_grad = True
                    params_to_optimize.append(param)
                else:
                    param.requires_grad = False
            # Add classifier params.
            for param in model.classifier.parameters():
                params_to_optimize.append(param)
            params_to_optimize = iter(params_to_optimize)
#        optimizer = optim.SGD(params_to_optimize, lr=args.lr,
 #                             momentum=0.9, weight_decay=args.weight_decay)
        optimizer = optim.Adam(params_to_optimize)
        # Perform finetuning.
        manager.train(args.finetune_epochs, optimizer,
                      save=True, savename=args.save_prefix)
    elif args.mode == 'check':
        # Load model and make sure everything is fine.
        manager.check(verbose=True)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        manager.eval_all()


if __name__ == '__main__':
    main()
