
# [ACL'25 Findings]MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models
  
<div>
<div align="center">
    <a href='https://z1zs.github.io/' target='_blank'>Jiahao Huo<sup>1,3</sup></a> 
    <a href='https://stupidbuluchacha.github.io/' target='_blank'>Yibo Yan<sup>1,2</sup></a> 
    <a href='https://zhengxujosh.github.io/' target='_blank'>Zheng Xu<sup>1,2</sup></a> 
    <a href='https://qc-ly.github.io/' target='_blank'>Yuanhuiyi Lyu<sup>1,2</sup></a> 
    <a href='https://scholar.google.com/citations?user=z39tx_sAAAAJ' target='_blank'>Xin Zou<sup>1,2</sup></a> 
    <a href='https://ieeexplore.ieee.org/author/37709584000' target='_blank'>Zhihua Wei<sup>3</sup></a> 
    <a href='https://xuminghu.github.io/' target='_blank'>Xuming Hu<sup>✉,1,2</sup></a> 
</div>
<div>
<div align="center">
    <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <br>   
    <sup>2</sup>The Hong Kong University of Science and Technology   
    <sup>3</sup>Tongji University <br>  
    <sup>✉</sup> Corresponding Author
</div>

---

Official implementation of "[MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](https://arxiv.org/abs/2502.11051)".  
Our codes are borrowed from [Liu](https://github.com/franciscoliu)'s baselines implementation [here](https://github.com/franciscoliu/MLLMU-Bench) and [Huang](https://github.com/K1nght)'s SRF-on implementation [here](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry). Thanks a lot for their efforts!

## Updates

- **16 Feb, 2025** :Paper published in Arxiv.
- **16 May, 2025** : Paper accepted by ACL 2025 as Findings.
- **21 May, 2025** : Code published.

---

This repository contains the **official implementation** of the following paper:

> **MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model** https://arxiv.org/abs/2502.11051
>
> **Abstract:** _Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to reformulate the task of multimodal MU in the era of MLLMs, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we develop a novel geometry-constrained gradient ascent method MMUnlearner. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code will be released upon acceptance._

## Get Start

- [Dataset](#download-dataset)
- [Vanilla Model](#getting-vanilla-model)
- [Running Baselines](#running-baselines)
- [Running MMUnlearner](#running-mmunlearner)
- [Evaluation](#running-mmunlearner)

  
## Download Dataset  
  
First, download the following datasets:  
  
- **MLLMU-Bench**: [MLLMU-Bench Dataset](https://huggingface.co/datasets/MLLMMU/MLLMU-Bench)  
- **CLEAR**: [CLEAR Dataset](https://huggingface.co/datasets/therem/CLEAR)  
  
Then, move them to the following directories:  
  
- `data/MLLMU-Bench`  
- `data/CLEAR`  

## Getting Vanilla Model  
  
To fine-tune the vanilla models, use the following commands:  
  
For MLLMU-Bench:  
  
```bash
python MLLMU_finetune.py --model_id path_to_original_model --forget_split_ratio 5 --save_dir path_to_mllmu_vanilla --batch_size 4 --lr 1e-5 --num_epochs 1 --data_split_dir data/MLLMMU-Bench
```
For CLEAR:  
```bash
python CLEAR_reference.py --model_id path_to_original_model --forget_ratio 5 --save_dir path_to_clear_vanilla --batch_size 4 --lr 1e-5 --num_epochs 1
```
## Running Baselines  
### GA  
To run the GA baseline:  
  
For MLLMU-Bench:  
```bash
python MLLMU_GA.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
For CLEAR:  
```bash
python CLEAR_GA.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_folder data/CLEAR --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
`ans_only`: When True, the loss is only calculated on answer tokens; otherwise, it will be calculated on all the text tokens.  
  
### GA_Diff
To run the GA_Diff baseline:  
  
For MLLMU-Bench:  
```bash
python MLLMU_GA_Diff.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
For CLEAR:  
```bash
python CLEAR_GA_Diff.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_folder data/CLEAR --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
### KL_Min  
To run the KL_Min baseline:  
  
For MLLMU-Bench:  
```bash
python MLLMU_KL_Min.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
For CLEAR:  
```bash
python CLEAR_KL_Min.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_folder data/CLEAR --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```
### NPO
First, get the reference model:  
  
For MLLMU-Bench:  
```bash
python MLLMU_reference.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --data_split_dir data/MLLMMU-Bench
```
For CLEAR:  
```bash 
python CLEAR_reference.py --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1
```
Then, run the NPO baseline:  
  
For MLLMU-Bench:  
```bash
python MLLMU_NPO.py --oracle_model_id path_to_ref_model --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --data_split_dir data/MLLMMU-Bench --ans_only True
```
For CLEAR:  
```bash
python CLEAR_NPO.py --oracle_model_id path_to_ref_model --model_id path_to_original_model --vanilla_dir path_to_vanilla_model --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --ans_only True
```

## Running MMUnlearner  
### Generate Gradient Mask  
To generate the gradient mask, run:  
```bash
cd data_process
python data_process/MLLMU_gen_mask.py
python data_process/CLEAR_gen_mask.py
```
### Selectively Unlearning  
To run the selective unlearning process:  
  
For MLLMU-Bench:  
```bash
python MLLMU_manifold.py --model_id path_to_original_model --data_split_dir data/MLLMMU-Bench --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --data_split_dir data/MLLMMU-Bench --grad_mask_path "path_to/mllmu_language_mask.pt" --ans_only True
```
For CLEAR:  
```bash
python CLEAR_manifold.py --model_id path_to_original_model --forget_split_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --grad_mask_path "path_to/clear_language_mask.pt" --ans_only True
```
`grad_mask_path`: Specifies the generated gradient mask, indicating which module(s) to update. Optional: language_mask.pt, vision_mask.pt, both_mask.pt.  
  
## Evaluation  
To evaluate the models, use the following commands:  
  
For MLLMU-Bench:  
```bash
bash MLLMU_eval.sh forget_ratio gpu_id1 gpu_id2 gpu_id3 gpu_id4 "path_to_evaluated_model" "path_to_original_model" "shot_num"
```
For CLEAR:  
```bash
bash CLEAR_eval.sh forget_ratio 100-forget_ratio "path_to_evaluated_model" "path_to_original_model" gpu_id1 gpu_id2 gpu_id3 gpu_id4
```
`gpu_id`: Specify the GPU to use (0-7).  
`forget_ratio`: Specify the forget ratio (e.g., 05 or 5).  
`shot_num`: Choose between "zero_shot" or "few_shot". For details, see MLLMU-Bench [Issue](https://github.com/franciscoliu/MLLMU-Bench/issues/2).  


