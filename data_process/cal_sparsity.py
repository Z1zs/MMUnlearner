
import torch
grad_mask_path="/data/jiahao/MMUnlearning/MLLMU-Bench/data_process/llava_mask/forget05/mllmu_both_mask.pt"
if grad_mask_path:
    module_set=set()
    grad_data=torch.load(grad_mask_path)
    grad_mask=grad_data['weight']
    layer_name_list=list(grad_mask.keys())
    for name in layer_name_list:
        if "proj" in name:
            continue
        elif "fc" in name:
            continue
        elif "linear" in name:
            continue
        elif "mlp" in name:
            continue
        elif "qkv" in name:
            continue
        else:
            grad_mask.pop(name)
    def calc_sparsity(tensor):
        num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)
        total_elements = tensor.numel()
        sparsity = num_zero_elements / total_elements
        return sparsity.item(), total_elements, num_zero_elements

    total_cnt=0
    w_cnt=0
    for k,v in grad_mask.items():
        try: 
            w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(v)
            total_cnt += total_elements
            w_cnt += w_num_zero_elements
            module_set=module_set|set(k.split("."))
        except: 
            pass 
    print("Saliency mask generated!")
    print(f"Total sparsity among weight:{w_cnt/total_cnt*100}")