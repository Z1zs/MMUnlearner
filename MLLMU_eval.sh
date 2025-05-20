#!/bin/bash
forget_ratio=$1
gpu_id1=$2
gpu_id2=$3
gpu_id3=$4
gpu_id4=$5
model_id=$7
cache_path=$6
test_data="data/MLLMU-Bench/Test_Set"
few_shot_data="data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet"
data_split_folder="data/MLLMU-Bench"
celebrity_data="data/MLLMU-Bench/Retain_Set/train-00000-of-00001.parquet"
shot_num=$8
output_name="output_name"
output_folder=${cache_path}/${shot_num}

CUDA_VISIBLE_DEVICES=$gpu_id1 python MLLMU_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "forget" --test_data ${test_data} --few_shot_data ${few_shot_data} --data_split_folder ${data_split_folder} --celebrity_data ${celebrity_data} --output_file ${output_name}-forget --output_folder ${output_folder}/forget${forget_ratio} --forget_ratio ${forget_ratio} --shot_num ${shot_num} &

CUDA_VISIBLE_DEVICES=$gpu_id2 python MLLMU_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "test" --test_data ${test_data} --few_shot_data ${few_shot_data} --data_split_folder ${data_split_folder} --celebrity_data ${celebrity_data} --output_file ${output_name}-test --output_folder ${output_folder}/forget${forget_ratio} --forget_ratio ${forget_ratio} --shot_num ${shot_num} &

CUDA_VISIBLE_DEVICES=$gpu_id3 python MLLMU_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "retain" --test_data ${test_data} --few_shot_data ${few_shot_data} --data_split_folder ${data_split_folder} --celebrity_data ${celebrity_data} --output_file ${output_name}-retain --output_folder ${output_folder}/forget${forget_ratio} --forget_ratio ${forget_ratio} --shot_num ${shot_num} &

CUDA_VISIBLE_DEVICES=$gpu_id4 python MLLMU_eval.py --model_id ${model_id} --cache_path ${cache_path} --eval_list "real" --test_data ${test_data} --few_shot_data ${few_shot_data} --data_split_folder ${data_split_folder} --celebrity_data ${celebrity_data} --output_file ${output_name}-real --output_folder ${output_folder}/forget${forget_ratio} --forget_ratio ${forget_ratio} --shot_num ${shot_num} &
