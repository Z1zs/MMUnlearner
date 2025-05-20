#!/bin/bash
forget_ratio=$1
retain_ratio=$2
model_id=$3
cache_path=$4
data_folder="data/CLEAR"
gpu_id1=$5
gpu_id2=$6
gpu_id3=$7
gpu_id4=$8

forget_cls_folder=forget${forget_ratio}_perturbed
forget_gen_folder=forget${forget_ratio}+tofu
retain_cls_folder=retain${retain_ratio}_perturbed
retain_gen_folder=retain${retain_ratio}+tofu
realface_folder=real_faces
realworld_folder=real_world
shot_num="zero_shots"
output_folder=${cache_path}/${shot_num}

CUDA_VISIBLE_DEVICES=$gpu_id1 python CLEAR_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "forget" --output_file llava-1.5-7b-clear-forget${forget_ratio} --output_folder ${output_folder}/forget${forget_ratio} --shot_num ${shot_num} --data_folder ${data_folder} --forget_cls_folder ${forget_cls_folder} --forget_gen_folder ${forget_gen_folder} --retain_cls_folder ${retain_cls_folder} --retain_gen_folder ${retain_gen_folder} --realface_folder ${realface_folder} --realworld_folder ${realworld_folder} &

CUDA_VISIBLE_DEVICES=$gpu_id2 python CLEAR_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "retain" --output_file llava-1.5-7b-clear-retain${retain_ratio} --output_folder ${output_folder}/forget${forget_ratio} --shot_num ${shot_num} --data_folder ${data_folder} --forget_cls_folder ${forget_cls_folder} --forget_gen_folder ${forget_gen_folder} --retain_cls_folder ${retain_cls_folder} --retain_gen_folder ${retain_gen_folder} --realface_folder ${realface_folder} --realworld_folder ${realworld_folder} &

CUDA_VISIBLE_DEVICES=$gpu_id3 python CLEAR_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "realface" --output_file llava-1.5-7b-clear-realface --output_folder ${output_folder}/forget${forget_ratio} --shot_num ${shot_num} --data_folder ${data_folder} --forget_cls_folder ${forget_cls_folder} --forget_gen_folder ${forget_gen_folder} --retain_cls_folder ${retain_cls_folder} --retain_gen_folder ${retain_gen_folder} --realface_folder ${realface_folder} --realworld_folder ${realworld_folder} &

CUDA_VISIBLE_DEVICES=$gpu_id4 python CLEAR_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "realworld" --output_file llava-1.5-7b-clear-realworld --output_folder ${output_folder}/forget${forget_ratio} --shot_num ${shot_num} --data_folder ${data_folder} --forget_cls_folder ${forget_cls_folder} --forget_gen_folder ${forget_gen_folder} --retain_cls_folder ${retain_cls_folder} --retain_gen_folder ${retain_gen_folder} --realface_folder ${realface_folder} --realworld_folder ${realworld_folder} &
