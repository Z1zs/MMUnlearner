from copy import deepcopy
import os
import math 
import time 

import torch 
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm
# import utils
# from .unlearn_method import UnlearnMethod
# from trainer import validate


def cycle(dl):
    while True:
        for data in dl:
            yield data

def calc_sparsity(tensor):
    # Count zero elements
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)

    # Total number of elements
    total_elements = tensor.numel()

    # Compute sparsity
    sparsity = num_zero_elements / total_elements
    return sparsity.item(), total_elements, num_zero_elements

class Mask_Our:
    def __init__(self, model,lr) -> None:
        self.weight_saliency_mask = {}
        self.lr=lr
        self.forget_gradients = {}
        self.preserve_gradients = {}
        self.model=model

    def prepare_weight_saliency_mask(self, modules, forget_loader, preserve_loader, threshold,save_path):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)

        # forget fisher 
        forget_fisher_path = os.path.join(save_path, "forget_fisher.pt")
        if os.path.exists(forget_fisher_path):
            forget_gradients = torch.load(forget_fisher_path)
        else:
            forget_gradients={}
            self.model.train()
            for name, param in self.model.named_parameters():
                if any(md_name in name for md_name in modules):
                    forget_gradients[name] = None
                else:
                    pass
            print("Module list of gradient dict is :\n",forget_gradients.keys())
            progress_bar = tqdm(forget_loader)
            print(len(progress_bar),len(forget_loader))
            for i, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = -outputs.loss
                optimizer.zero_grad()
                loss.backward()
                if torch.isnan(loss):
                    print("WTF! Loss is NaN!!!")

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if any(md_name in name for md_name in modules) and param.grad is not None:
                            if forget_gradients[name] is not None:
                                forget_gradients[name] += param.grad.data.cpu()**2 / len(forget_loader)
                            else:
                                forget_gradients[name] = param.grad.data.cpu()**2 / len(forget_loader)
                # break
            for k,v in forget_gradients.items():
                self.forget_gradients[k]=v
        print("Forget mainfold finished!")


        # preserve fisher 
        preserve_fisher_path = os.path.join(save_path, "preserve_fisher.pt")
        if os.path.exists(preserve_fisher_path):
            preserve_gradients = torch.load(preserve_fisher_path)
        else:
            preserve_gradients={}
            self.model.train()
            for name, param in self.model.named_parameters():
                if any(md_name in name for md_name in modules):
                    preserve_gradients[name] = None
                else:
                    pass
            
            progress_bar = tqdm(preserve_loader)
            print(len(progress_bar),len(preserve_loader))
            for i, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = -outputs.loss
                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if any(md_name in name for md_name in modules) and param.grad is not None:
                            if preserve_gradients[name] is not None:
                                preserve_gradients[name] += param.grad.data.cpu()**2 / len(preserve_loader)
                            else:
                                preserve_gradients[name] = param.grad.data.cpu()**2 / len(preserve_loader)
                # break
            for k,v in preserve_gradients.items():
                self.preserve_gradients[k]=v
        print("Preserve mainfold finished!")


        total_cnt = 0 
        w_cnt = 0 
        weight_saliency_mask = {}
        for name in forget_gradients.keys():
            weight_saliency_mask[name] = 0
            try: 
                weight_saliency = (forget_gradients[name] + 1e-15) / (preserve_gradients[name] + 1e-15)
                w = weight_saliency >= threshold
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                weight_saliency_mask[name] = w
            except: 
                pass 
        for k,v in weight_saliency_mask.items():
                self.weight_saliency_mask[k]=v
        print("Saliency mask generated!")
        print(f"Total sparsity th@{threshold} among {modules}'s weight:{w_cnt/total_cnt*100}")
        return weight_saliency_mask,forget_gradients,preserve_gradients

    def get_weight_saliency_mask(self):
        return self.weight_saliency_mask,self.forget_gradients,self.preserve_gradients
    
    
class Mask_grad:
    def __init__(self, model,lr) -> None:
        self.weight_saliency_mask = {}
        self.lr=lr
        self.forget_gradients = {}
        self.preserve_gradients = {}
        self.model=model

    def prepare_weight_saliency_mask(self, modules, forget_loader, preserve_loader, threshold,save_path):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)

        # forget fisher 
        forget_fisher_path = os.path.join(save_path, "forget_fisher.pt")
        if os.path.exists(forget_fisher_path):
            forget_gradients = torch.load(forget_fisher_path)
        else:
            forget_gradients={}
            self.model.train()
            for name, param in self.model.named_parameters():
                if any(md_name in name for md_name in modules):
                    forget_gradients[name] = None
                else:
                    pass
            print("Module list of gradient dict is :\n",forget_gradients.keys())
            progress_bar = tqdm(forget_loader)
            print(len(progress_bar),len(forget_loader))
            for i, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = -outputs.loss
                optimizer.zero_grad()
                loss.backward()
                if torch.isnan(loss):
                    print("WTF! Loss is NaN!!!")

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if any(md_name in name for md_name in modules) and param.grad is not None:
                            if forget_gradients[name] is not None:
                                forget_gradients[name] += param.grad.data.cpu().abs() / len(forget_loader)
                            else:
                                forget_gradients[name] = param.grad.data.cpu().abs() / len(forget_loader)
                # break
            for k,v in forget_gradients.items():
                self.forget_gradients[k]=v
        print("Forget mainfold finished!")


        # preserve fisher 
        preserve_fisher_path = os.path.join(save_path, "preserve_fisher.pt")
        if os.path.exists(preserve_fisher_path):
            preserve_gradients = torch.load(preserve_fisher_path)
        else:
            preserve_gradients={}
            self.model.train()
            for name, param in self.model.named_parameters():
                if any(md_name in name for md_name in modules):
                    preserve_gradients[name] = None
                else:
                    pass
            
            progress_bar = tqdm(preserve_loader)
            print(len(progress_bar),len(preserve_loader))
            for i, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = -outputs.loss
                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if any(md_name in name for md_name in modules) and param.grad is not None:
                            if preserve_gradients[name] is not None:
                                preserve_gradients[name] += param.grad.data.cpu().abs() / len(preserve_loader)
                            else:
                                preserve_gradients[name] = param.grad.data.cpu().abs() / len(preserve_loader)
                # break
            for k,v in preserve_gradients.items():
                self.preserve_gradients[k]=v
        print("Preserve mainfold finished!")


        total_cnt = 0 
        w_cnt = 0 
        weight_saliency_mask = {}
        for name in forget_gradients.keys():
            weight_saliency_mask[name] = 0
            try: 
                weight_saliency1 = (forget_gradients[name] + 1e-15) / (preserve_gradients[name] + 1e-15)
                weight_saliency2 = (forget_gradients[name] + 1e-15) 
                w1 = weight_saliency1 >= threshold
                
                k = w1.sum().item()
                topk_values, topk_indices = torch.topk(weight_saliency2.view(-1), k)
                w2 = torch.zeros_like(weight_saliency2, dtype=torch.bool)
                w2.view(-1)[topk_indices] = True
                
                
                num_positions = weight_saliency2.numel()
                random_indices = torch.randperm(num_positions, device=weight_saliency2.device)
                selected_random_indices = random_indices[:k]
                w3 = torch.zeros_like(weight_saliency2, dtype=torch.bool)
                w3.view(-1)[selected_random_indices] = True
                
                w=w3
                
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                weight_saliency_mask[name] = w
            except: 
                pass 
        for k,v in weight_saliency_mask.items():
                self.weight_saliency_mask[k]=v
        print("Saliency mask generated!")
        print(f"Total sparsity th@{threshold} among {modules}'s weight:{w_cnt/total_cnt*100}")
        return weight_saliency_mask,forget_gradients,preserve_gradients

    def get_weight_saliency_mask(self):
        return self.weight_saliency_mask,self.forget_gradients,self.preserve_gradients
