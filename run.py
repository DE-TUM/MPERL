import time
import os, glob
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from src.models import PonderRelationalGraphConvModel, ReconstructionLoss, RegularizationLoss
from src.data_utils import load_data, load_data_yago_fb
from src.utils import row_normalize, accuracy, f1_macro, get_splits, evaluate
from src.params import args
from src.pytorchtools import EarlyStopping
import pandas as pd
from random import randrange
warnings.filterwarnings('ignore')


from src.losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from src.helpers import one_hot_embedding

class Train:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        self.best_val = 0
        self.batch_size = self.args.batch_size

        if self.args.data in ['YAGO43k', 'FB15k']:
            # self.y -> all_true, self.idx_train -> array
            # y_train -> shape(n_entities, n_classes)
            self.A, self.y, self.idx_train, self.idx_val, self.idx_test, y_train, y_valid, y_test = self.input_data_fb_yago()
            self.num_nodes = self.A[0].shape[0]
            self.num_rel = len(self.A)
            self.labels = torch.tensor(self.y.toarray(), dtype=torch.long) # shape(n_entities, n_classes), all_true
            self.y_train = torch.tensor(y_train.toarray(), dtype=torch.long) # shape(n_entities, n_classes)
            self.y_val = torch.tensor(y_valid.toarray(), dtype=torch.long)
            self.y_test = torch.tensor(y_test.toarray(), dtype=torch.long)
        else:

            # Load data
            self.A, self.y, self.train_idx, self.test_idx = self.input_data()
            self.num_nodes = self.A[0].shape[0]
            self.num_rel = len(self.A)
            self.labels = torch.LongTensor(np.array(np.argmax(self.y, axis=-1)).squeeze()) # shape(n_entities,)
            
            # Get dataset splits
            self.y_train, self.y_val, self.y_test, self.idx_train, self.idx_val, self.idx_test = get_splits(self.y, self.train_idx, self.test_idx, self.args.validation)

        # Adjacency matrix normalization
        self.A = row_normalize(self.A)
        
        # Create Model
        # Yiwen: adaptable max markov steps
        self.lambda_p = args.lambda_p
        self.max_steps = int(1/self.lambda_p) #+ 1 # +1 or not -> try both
        self.model = PonderRelationalGraphConvModel(input_size=self.num_nodes, hidden_size=self.args.hidden, output_size=self.y_train.shape[1], num_bases=self.args.bases, num_rel=self.num_rel, num_layer=2, dropout=self.args.drop, max_steps=self.max_steps, featureless=True, cuda=self.args.using_cuda, seed=self.args.seed)
        print('Loaded %s dataset with %d entities, %d relations and %d classes' % (self.args.data, self.num_nodes, self.num_rel / 2, self.y_train.shape[1]))
        
        # Loss and optimizer
        self.loss_rec = edl_digamma_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.steplr, gamma=0.2)

        # $L_{Reg}$
        self.loss_reg = RegularizationLoss(self.lambda_p, self.args.epochs)#.to(self.device)
        self.beta = 0.01

        # initialize the early_stopping object
        if self.args.validation:
            self.early_stopping = EarlyStopping(patience=10, verbose=True)

        if self.args.using_cuda:
            print("Using the GPU")
            self.model.cuda()
            # self.labels = self.labels.cuda()
        
      
    def input_data(self, dirname='./data'):
        data = None
        if os.path.isfile(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle'):
            with open(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle', 'rb') as f:
                data = pkl.load(f)
        else:
            with open(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle', 'wb') as f:
                # Data Loading...    
                A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names = load_data(self.args.data, self.args.hop)
                data = {
                    'A': A,
                    'y': y,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                }
                pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        return data['A'], data['y'], data['train_idx'], data['test_idx']
    
    def train(self, epoch):  
        t = time.time()
        X = None # featureless
        # Start training
        self.model.train()
        emb_train, p, lamda = self.model(A=self.A, X=None)
        tmp = p[:,self.idx_train] # shape(#steps, #train_data)

        # # Yiwen: 2024-09-17 loss calculation for each step and then sum up -> not working
        # y = one_hot_embedding(self.labels[self.idx_train], self.y.shape[1])
        # if self.args.using_cuda:
        #     self.labels = self.labels.cuda()
        # loss = 0
        # n_steps = p.shape[0]
        # for j in range(1, n_steps):
        #     # h = \sum_{s=1}^{n} h_s* \lambda_s 
        #     emb_train_j = torch.sum(emb_train[j-1:j] * lamda[j-1:j].unsqueeze(-1), dim=0)
        #     loss_rec = self.loss_rec(emb_train_j[self.idx_train], y.float(), epoch, self.y.shape[1], 10*y.shape[1], torch.mean(tmp[j-1:j]), 0)

        #     # $L = L_{Rec} + \beta L_{Reg}$
        #     loss_j = loss_rec + self.beta * self.loss_reg(tmp[j-1:j], self.args.using_cuda)
        #     loss += loss_j
        
        # Calculate the regularization loss
        loss_reg = self.loss_reg(tmp, self.args.using_cuda)

        if self.args.using_cuda:
            self.labels = self.labels.cuda()

        # Yiwen: To follow paper h = \sum_{s=1}^{n} h_s* \lambda_s 
        emb_train = torch.sum(emb_train * lamda.unsqueeze(-1), dim=0) # [num_nodes, output_size]
        # emb_train = torch.mean(emb_train, 0) # assume lambda_n = lambda_p
        y = one_hot_embedding(self.labels[self.idx_train], self.y.shape[1])
        loss_rec = self.loss_rec(emb_train[self.idx_train], y.float(), epoch, self.y.shape[1], 10*y.shape[1], torch.mean(tmp), 0)

        # $L = L_{Rec} + \beta L_{Reg}$
        loss = loss_rec + self.beta * loss_reg
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        acc = accuracy(emb_train[self.idx_train], self.labels[self.idx_train]).item()*100
        # test data
        tmp = p[:,self.idx_test]
        y = one_hot_embedding(self.labels[self.idx_test], self.y.shape[1])
        loss_test = self.loss_rec(emb_train[self.idx_test], y.float(), epoch, self.y.shape[1], 10*y.shape[1], torch.mean(tmp), 0) + self.beta * self.loss_reg(tmp, self.args.using_cuda)
        acc_test = accuracy(emb_train[self.idx_test], self.labels[self.idx_test]).item()*100
        f1_test = f1_macro(emb_train[self.idx_test], self.labels[self.idx_test]).item()*100
        loss_test = loss_test.item()

        print ("Epoch: {epoch}, Training Loss on {num} training data: {loss}, Training Accuracy: {acc_train}, ####### Testing Accuracy: {acc_test}, Testing F1_Macro: {f1_test}".format(epoch=epoch, num=len(self.idx_train), loss=str(loss.item()), acc_test=acc_test, acc_train=acc, f1_test=f1_test))

        if self.args.validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            with torch.no_grad():
                self.model.eval()
                emb_valid, p, _ = self.model(A=self.A, X=None)
                tmp = p[:,self.idx_val]

                emb_valid = torch.mean(emb_valid, 0)
                y = one_hot_embedding(self.labels[self.idx_val], self.y.shape[1])
                loss_val = self.loss_rec(emb_valid[self.idx_val], y.float(), epoch, self.y.shape[1], 10*y.shape[1], torch.mean(tmp), 0) + self.beta * self.loss_reg(tmp, self.args.using_cuda)
                acc_val = accuracy(emb_valid[self.idx_val], self.labels[self.idx_val])
                if acc_val >= self.best_val:
                    self.best_val = acc_val
                    self.model_state = {
                        'state_dict': self.model.state_dict(),
                        'best_val': acc_val,
                        'best_epoch': epoch,
                        'optimizer': self.optimizer.state_dict(),
                    }
                print('loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
                print('\n')
                    
                self.early_stopping(loss_val, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.model_state = {
                        'state_dict': self.model.state_dict(),
                        'best_val': acc_val,
                        'best_epoch': epoch,
                        'optimizer': self.optimizer.state_dict(),
                    }
                    return False
        return True
            
    def test(self):
        with torch.no_grad():
            self.model.eval()
            emb_test, p, _ = self.model(A=self.A, X=None)
            tmp = p[:,self.idx_test]
            loss_reg = self.loss_reg(tmp, self.args.using_cuda)

            emb_test = torch.mean(emb_test, 0)
            y = one_hot_embedding(self.labels[self.idx_test], self.y.shape[1])
            loss_rec = self.loss_rec(emb_test[self.idx_test], y.float(), epoch, self.y.shape[1], 10*y.shape[1], torch.mean(tmp), 0)

            # $L = L_{Rec} + \beta L_{Reg}$
            loss_test = loss_rec + self.beta * loss_reg
            loss_test = loss_test.item()
            acc_test = accuracy(emb_test[self.idx_test], self.labels[self.idx_test]).item()
            f1_test = f1_macro(emb_test[self.idx_test], self.labels[self.idx_test]).item()
            print('Accuracy of the network on the {num} test data: {acc} %, F1_Macro {f1} %, loss: {loss}'.format(num=len(self.idx_test), acc=acc_test*100, f1=f1_test*100, loss=loss_test)) 
        return loss_test, acc_test*100
    

    def save_checkpoint(self, filename='./.checkpoints/'+args.name):
        print('Save model...')
        if not os.path.exists('.checkpoints'):
            os.makedirs('.checkpoints')
        torch.save(self.model_state, filename)
        print('Successfully saved model\n...')

    def load_checkpoint(self, filename='./.checkpoints/'+args.name, ts='teacher'):
        print('Load model...')
        load_state = torch.load(filename)
        self.model.load_state_dict(load_state['state_dict'])
        self.optimizer.load_state_dict(load_state['optimizer'])
        print('Successfully Loaded model\n...')
        print("Best Epoch:", load_state['best_epoch'])
        print("Best acc_val:", load_state['best_val'].item())

    # added by yiwen
    def input_data_fb_yago(self, dirname='./data'):
        data = None
        if os.path.isfile(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle'):
            with open(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle', 'rb') as f:
                data = pkl.load(f)
        else:
            with open(dirname + '/' + self.args.data + '_' + str(self.args.hop) + '.pickle', 'wb') as f:
                # Data Loading...    
                A, X, y, train_idx, valid_idx, test_idx, rel_dict, train_names, valid_names, test_names, train_labels, valid_labels, test_labels = load_data_yago_fb(self.args.data, self.args.hop)
                data = {
                    'A': A,
                    'y': y,
                    'train_idx': train_idx,
                    'valid_idx': valid_idx,
                    'test_idx': test_idx,
                    'train_labels': train_labels,
                    'valid_labels': valid_labels,
                    'test_labels': test_labels,
                }
                pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        return data['A'], data['y'], data['train_idx'], data['valid_idx'], data['test_idx'], data['train_labels'], data['valid_labels'], data['test_labels']
    

    def train_fb_yago(self, epoch):  
        t = time.time()
        X = None # featureless
        # Start training
        self.model.train()

        # shuffle the training data
        np.random.shuffle(self.idx_train)
        log = []
        for i in range(0, len(self.idx_train), self.batch_size):
            batch_idx = self.idx_train[i:min(i+self.batch_size, len(self.idx_train))]
            batch = self.y_train[batch_idx]
            
            emb_train, p = self.model(A=self.A, X=None)
            emb_train = torch.mean(emb_train, 0)

            if args.using_cuda:
                batch = batch.cuda()
                # batch_idx = batch_idx.cuda()
            tmp = p[:,batch_idx]

            # Calculate the regularization loss and reconstruction loss
            loss_reg = self.loss_reg(tmp, self.args.using_cuda)
            loss_rec = self.loss_rec(emb_train[batch_idx], batch.float(), epoch, self.y.shape[1], 10*batch.shape[1], torch.mean(tmp), 0)

            # $L = L_{Rec} + \beta L_{Reg}$
            loss = loss_rec + self.beta * loss_reg

            # Log the loss
            log.append(loss.item())
        
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Calculate the average loss
        avg_loss = np.mean(log)
        print("Epoch: {epoch}, Training Loss on {num} training data: {loss}, time:{time}".format(epoch=epoch, num=len(self.idx_train), loss=str(avg_loss), time=time.time()-t))

        if self.args.validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            with torch.no_grad():
                self.model.eval()
                emb_valid, p = self.model(A=self.A, X=None)
                emb_valid = torch.mean(emb_valid.cpu(), 0)
                tmp = p[:,self.idx_val]
                loss_reg = self.loss_reg(tmp, self.args.using_cuda).cpu()
                loss_val = self.loss_rec(emb_valid[self.idx_val], self.y_val[self.idx_val].float(), epoch, self.y.shape[1], 10*self.y_val.shape[1], torch.mean(tmp.cpu()), 0) + self.beta * loss_reg
                mrr_val, hit1_val, hit3_val, hit10_val = evaluate(emb_valid[self.idx_val], self.y_val[self.idx_val], self.labels[self.idx_val])
                
                self.early_stopping(-mrr_val, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    # self.model_state = {
                    #     'state_dict': self.model.state_dict(),
                    #     'best_val': mrr_val,
                    #     'best_epoch': epoch,
                    #     'optimizer': self.optimizer.state_dict(),
                    # }
                    return False
                else:
                    if self.early_stopping.counter == 0:
                        torch.save(self.model.state_dict(), 'best_model.pkl')
                        self.model_state = {
                            'state_dict': self.model.state_dict(),
                            'best_val': mrr_val,
                            'best_epoch': epoch,
                            'optimizer': self.optimizer.state_dict(),
                        }
                    print('loss_val: {:.4f}'.format(loss_val.item()),
                    'mrr_val: {:.4f}'.format(mrr_val.item()),
                    'time: {:.4f}s'.format(time.time() - t))
                    print('\n')
        return True
    

    def test_fb_yago(self):
        # load the best model
        self.model.load_state_dict(torch.load('best_model.pkl'))
        self.model.eval()
        with torch.no_grad():
            emb_test, p = self.model(A=self.A, X=None)
            emb_test = torch.mean(emb_test.cpu(), 0)
            tmp = p[:,self.idx_test]
            loss_reg = self.loss_reg(tmp, self.args.using_cuda).cpu()
            loss_test = self.loss_rec(emb_test[self.idx_test], self.y_test[self.idx_test].float(), epoch, self.y.shape[1], 10*self.y_test.shape[1], torch.mean(tmp.cpu()), 0) + self.beta * loss_reg
            mrr_test, hit1_test, hit3_test, hit10_test = evaluate(emb_test[self.idx_test], self.y_test[self.idx_test], self.labels[self.idx_test])
            print('On the {num} test data: MRR: {mrr}, Hit@1:{hit1}, Hit@3:{hit3}, Hit@10:{hit10}'.format(num = len(self.idx_test), 
                                                                                                          mrr = mrr_test.item(), 
                                                                                                          hit1 = hit1_test.item(), 
                                                                                                          hit3 = hit3_test.item(), 
                                                                                                          hit10 = hit10_test.item()))
        return loss_test, mrr_test, hit1_test, hit3_test, hit10_test
        
        
if __name__ == '__main__':
    train = Train(args)

    if args.data in ['YAGO43k', 'FB15k']:
        for epoch in range(args.epochs):
            if train.train_fb_yago(epoch) is False:
                break
    else:
        for epoch in range(args.epochs):
            # # Debugging for memory leak
            # with torch.autograd.detect_anomaly():
            # with profile(activities=[ProfilerActivity.CPU],
            #              profile_memory=True, record_shapes=True) as prof:
            if train.train(epoch) is False:
                break
            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    if args.validation:
        train.save_checkpoint()
        train.load_checkpoint()

    if args.data in ['YAGO43k', 'FB15k']:
        train.test_fb_yago()
    else:
        train.test()

