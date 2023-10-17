import numpy as np
import scipy as scp
import pandas as pd
import torch

from tqdm import tqdm
import os

from utils_captum import *

ALL_DEVICES = ['cuda:0',
               'cuda:1',
               'cuda:2',
               'cuda:3',
              ]

def trainer(net, loss_fn, extract_fn, optimizer, data_loaders, n_epoch=10, valid_iter=1, test_iter=1, device='cpu',save_path='',fname='net',save_file=True):

    # Set current validation loss as best validation loss.
    net.eval()
    best_valid_loss = loss_fn(net,data_loaders['valid'],device).detach().cpu()
    test_valid_loss = loss_fn(net,data_loaders['test'],device).detach().cpu()
    best_state = extract_fn(net)
    
    # Save validation losses
    train_loss_list = []
    valid_loss_list = [best_valid_loss]
    test_loss_list = [test_valid_loss]
        
    n_length = len(str(n_epoch))
    schedule_count = 0

    for n in range(1,n_epoch+1):
        print(f'Epoch {n}:')
        
        # Start training
        net.train()
        print('Training...')
        optimizer.zero_grad()
        train_loss = loss_fn(net, data_loaders['train'],device)
        train_loss.backward()
        optimizer.step()
        
        print(f'Train loss: {train_loss}')
        train_loss_list.append(train_loss.detach().cpu().item())
        del train_loss
        
        net.eval()        
        if n % valid_iter == 0:
            print('Validation...')
            valid_loss = loss_fn(net, data_loaders['valid'],device)
            print(f'Validation loss: {valid_loss}')
            valid_loss_list.append(valid_loss.detach().cpu().item())
            
            if valid_loss < best_valid_loss:
                if save_file:
                    save_name = save_path+fname+f'_epoch_{str(n).zfill(n_length)}.pt'
                    torch.save(net.state_dict(),save_name)
                best_state = extract_fn(net)
                best_valid_loss = valid_loss
                print('Updated.')                              
            del valid_loss

        if n % test_iter == 0:
            print('Test...')
            test_loss = loss_fn(net, data_loaders['test'],device)
            print(f'Test loss: {test_loss}')
            test_loss_list.append(test_loss.detach().cpu().item())
            
            del test_loss
        
    final_state = extract_fn(net)
        
    return train_loss_list, valid_loss_list, test_loss_list, best_state, final_state


def trainer_loop(net, loss_fn, extract_fn, optimizer, data_loaders, n_epoch=10, valid_iter=1, test_iter=1, max_iter=4, device='cpu',save_path='',fname='net',save_file=True):

    net.eval()
    
    # Save validation losses
    train_loss_list = []
    total_epoch = 0
    best_state = extract_fn(net)
    
    # Look at save location to find previous best state, if any.
    # Save that model's iteration.
    prev_models = os.listdir(save_path)
    prev_models = [p for p in prev_models if fname in p]
    prev_models.sort()
    if len(prev_models) > 0:
        best_prev = prev_models[-1]
        total_epoch = int(best_prev.split('.')[0].split('_')[-1])
        best_state = torch.load(best_prev)
        net.load_state_dict(best_state)    

    n_length = max(len(str(n_epoch)), len(str(total_epoch)))
    
    # Set current validation loss as best validation loss.    
    if valid_iter is not None:
        best_valid_loss = 0
        for inputs in tqdm(iter(data_loaders['valid'])):
            best_valid_loss += loss_fn(net, inputs, device).detach().cpu().item()
        best_valid_loss /= len(data_loaders['valid'].dataset)
        print(f'Validation loss: {best_valid_loss}')
        valid_loss_list = [best_valid_loss]
    else:
        valid_loss_list = []

    if test_iter is not None:
        test_loss = 0
        for inputs in tqdm(iter(data_loaders['test'])):
            test_loss += loss_fn(net, inputs, device).detach().cpu().item()
        test_loss /= len(data_loaders['test'].dataset)
        print(f'Test loss: {test_loss}')    
        test_loss_list = [test_loss] 
    else:
        test_loss_list = []
        
    fail_count = 0
    for n in range(1,n_epoch+1):
        total_epoch += 1
        print(f'Epoch {total_epoch}:')
        
        # Start training
        net.train()
        print('Training...')
        tr_loss = 0
        for inputs in tqdm(iter(data_loaders['train'])):
            optimizer.zero_grad()
            train_loss = loss_fn(net, inputs, device)
            train_loss.backward()
            optimizer.step()
            tr_loss += train_loss.detach().cpu().item()
            del train_loss
        tr_loss /= len(data_loaders['train'].dataset)
        print(f'Train loss: {tr_loss}')
        train_loss_list.append(tr_loss)
        del tr_loss

        net.eval()
        if valid_iter is not None:
            if total_epoch % valid_iter == 0:
                valid_loss = 0
                print('Validation...')
                for inputs in tqdm(iter(data_loaders['valid'])):
                    valid_loss += loss_fn(net, inputs, device).detach().cpu().item()
                valid_loss /= len(data_loaders['valid'].dataset)
                print(f'Validation loss: {valid_loss}')
                valid_loss_list.append(valid_loss)

                if valid_loss < best_valid_loss:
                    if save_file:  
                        save_name = save_path+fname+f'_epoch_{str(total_epoch).zfill(n_length)}.pt'
                        torch.save(net.state_dict(),save_name)
                    best_state = extract_fn(net)
                    best_valid_loss = valid_loss
                    print('Updated.')     
                else:
                    fail_count += 1
                    if fail_count >= max_iter:
                        break
                del valid_loss            
                
        if test_iter is not None:
            if n % test_iter == 0:
                print('Test...')
                test_loss = 0
                for inputs in tqdm(iter(data_loaders['test'])):
                    test_loss += loss_fn(net, inputs, device).detach().cpu().item()
                test_loss /= len(data_loaders['test'].dataset)
                print(f'Test loss: {test_loss}')
                test_loss_list.append(test_loss)
                del test_loss
            
    final_state = extract_fn(net)
        
    return train_loss_list, valid_loss_list, test_loss_list, best_state, final_state


def trainer_iter(net, loss_fn, extract_fn, optimizer, data_loaders, n_epoch=10, max_iter=100000, valid_iter=1000, test_iter=1000, reset_iter=1000, device='cpu',save_path='',fname='net',save_file=True):
    
    # Set current validation loss as best validation loss.
    net.eval()
    if valid_iter is not None:
        best_valid_loss = 0
        for inputs in tqdm(iter(data_loaders['valid'])):
            best_valid_loss += loss_fn(net, inputs, device).detach().cpu().item()
        best_valid_loss /= len(data_loaders['valid'].dataset)
        print(f'Validation loss: {best_valid_loss}')
        valid_loss_list = [best_valid_loss]
    else:
        valid_loss_list = []

    if test_iter is not None:
        test_loss = 0
        for inputs in tqdm(iter(data_loaders['test'])):
            test_loss += loss_fn(net, inputs, device).detach().cpu().item()
        test_loss /= len(data_loaders['test'].dataset)
        print(f'Test loss: {test_loss}')    
        test_loss_list = [test_loss]
    else:
        test_loss_list = []
    
    # best_valid_loss = np.inf
    # test_loss = np.inf
    
    best_state = extract_fn(net)
    
    # Save validation losses
    train_loss_list = []
            
    n_length = len(str(max_iter))
    schedule_count = 0
    current_iter = 0
    total_iter = 0
    
    # Look at save location to find previous best state, if any.
    # Save that model's iteration.
    prev_models = os.listdir(save_path)
    prev_models = [p for p in prev_models if fname in p]
    prev_models.sort()
    if len(prev_models) > 0:
        best_prev = prev_models[-1]
        prev_iter = int(best_prev.split('.')[0].split('_')[-1])
        total_iter = prev_iter
        best_state = torch.load(best_prev)
        net.load_state_dict(best_state)
    
    for n in range(1,n_epoch+1):
        print(f'Epoch {n}:')
        tr_loss = 0
        net.train()
        for inputs in iter(data_loaders['train']):
            optimizer.zero_grad()
            train_loss = loss_fn(net, inputs, device)
            train_loss.backward()
            optimizer.step()
            tr_loss += train_loss.detach().cpu().item()*inputs.shape[0]
            current_iter += inputs.shape[0]
            total_iter += inputs.shape[0]
            del train_loss
            if current_iter > reset_iter:
                tr_loss /= current_iter
                print(f'Iter: {total_iter} / Train loss: {tr_loss}')
                train_loss_list.append(tr_loss)
                del tr_loss        
                tr_loss = 0
                
            net.eval()     
            if valid_iter is not None:
                if current_iter > valid_iter:
                    valid_loss = 0
                    for inputs in iter(data_loaders['valid']):
                        valid_loss += loss_fn(net, inputs, device)
                    valid_loss /= len(data_loaders['valid'].dataset)
                    print(f'Iter: {total_iter} / Validation loss: {valid_loss}')
                    valid_loss_list.append(valid_loss.detach().cpu().item())

                    if valid_loss < best_valid_loss:
                        if save_file:
                            save_name = save_path+fname+f'_iter_{str(total_iter).zfill(n_length)}.pt'
                            torch.save(net.state_dict(),save_name)
                        best_state = extract_fn(net)
                        best_valid_loss = valid_loss
                        print('Updated.')                              
                    del valid_loss

            if test_iter is not None:
                if current_iter > test_iter:
                    test_loss = 0
                    for inputs in iter(data_loaders['test']):
                        test_loss, _ = loss_fn(net, inputs, device)
                    test_loss /= len(data_loaders['test'].dataset)
                    print(f'Iter: {total_iter} / Test loss: {test_loss}')
                    test_loss_list.append(test_loss.detach().cpu().item())
                    del test_loss
        
        if current_iter > reset_iter:            
            current_iter = 0
        
        if total_iter >= max_iter:
            break
        
    final_state = extract_fn(net)
        
    return train_loss_list, valid_loss_list, test_loss_list, best_state, final_state

def trainer_loop_kwargs(net, loss_fn, extract_fn, optimizer, data_loaders, n_epoch=10, valid_iter=1, test_iter=1, device='cpu',save_path='',fname='net',save_file=True, **kwargs):

    # Set current validation loss as best validation loss.
    net.eval()
    if valid_iter is not None:
        best_valid_loss = 0
        for inputs in tqdm(iter(data_loaders['valid'])):
            best_valid_loss += loss_fn(net, inputs, device, kwargs).detach().cpu().item()
        best_valid_loss /= len(data_loaders['valid'].dataset)
        print(f'Validation loss: {best_valid_loss}')
        valid_loss_list = [best_valid_loss]
    else:
        valid_loss_list = []

    if test_iter is not None:
        test_loss = 0
        for inputs in tqdm(iter(data_loaders['test'])):
            test_loss += loss_fn(net, inputs, device, kwargs).detach().cpu().item()
        test_loss /= len(data_loaders['test'].dataset)
        print(f'Test loss: {test_loss}')    
        test_loss_list = [test_loss] 
    else:
        test_loss_list = []
        
    # Save validation losses
    train_loss_list = []
    total_epoch = 0
    best_state = extract_fn(net)
    
    # Look at save location to find previous best state, if any.
    # Save that model's iteration.
    prev_models = os.listdir(save_path)
    prev_models = [p for p in prev_models if fname in p]
    prev_models.sort()
    if len(prev_models) > 0:
        best_prev = prev_models[-1]
        total_epoch = int(best_prev.split('.')[0].split('_')[-1])
        best_state = torch.load(best_prev)
        net.load_state_dict(best_state)    

    n_length = max(len(str(n_epoch)), len(str(total_epoch)))
        
    for n in range(1,n_epoch+1):
        total_epoch += 1
        print(f'Epoch {total_epoch}:')
        
        # Start training
        net.train()
        print('Training...')
        tr_loss = 0
        for inputs in tqdm(iter(data_loaders['train'])):
            optimizer.zero_grad()
            train_loss = loss_fn(net, inputs, device, kwargs)
            train_loss.backward()
            optimizer.step()
            tr_loss += train_loss.detach().cpu().item()
            del train_loss
        tr_loss /= len(data_loaders['train'].dataset)
        print(f'Train loss: {tr_loss}')
        train_loss_list.append(tr_loss)
        del tr_loss

        net.eval()
        if valid_iter is not None:
            if total_epoch % valid_iter == 0:
                valid_loss = 0
                print('Validation...')
                for inputs in tqdm(iter(data_loaders['valid'])):
                    valid_loss += loss_fn(net, inputs, device, kwargs).detach().cpu().item()
                valid_loss /= len(data_loaders['valid'].dataset)
                print(f'Validation loss: {valid_loss}')
                valid_loss_list.append(valid_loss)

                if valid_loss < best_valid_loss:
                    if save_file:  
                        save_name = save_path+fname+f'_epoch_{str(total_epoch).zfill(n_length)}.pt'
                        torch.save(net.state_dict(),save_name)
                    best_state = extract_fn(net)
                    best_valid_loss = valid_loss
                    print('Updated.')        
                del valid_loss            
                
        if test_iter is not None:
            if n % test_iter == 0:
                print('Test...')
                test_loss = 0
                for inputs in tqdm(iter(data_loaders['test'])):
                    test_loss += loss_fn(net, inputs, device, kwargs).detach().cpu().item()
                test_loss /= len(data_loaders['test'].dataset)
                print(f'Test loss: {test_loss}')
                test_loss_list.append(test_loss)
                del test_loss
        
    final_state = extract_fn(net)
        
    return train_loss_list, valid_loss_list, test_loss_list, best_state, final_state