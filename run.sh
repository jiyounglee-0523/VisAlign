#!/bin/bash
for seed in "46 47 48 49"
do for model_name in convnext_extra swin_extra vit_30_16
  do for postprocessor_name in odin mds knn tapudd mcdropout
    do OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 python test_main.py --ckpt_dir /home/edlab/shkim/RELIABLE/reliable_project/outputs/seed$seed_$model_name/best.ckpt --seed $seed --model_name $model_name --test_id_dataset_path /home/edlab/jylee/RELIABLE/data/final_reatention --ood_adversarial_path /home/edlab/jylee/RELIABLE/data/adversarial --train_id_train_dataset_path /home/edlab/jylee/RELIABLE/data/final_dataset/final_train --train_id_eval_dataset_path /home/edlab/jylee/RELIABLE/data/final_dataset/final_eval --postprocessor_name $postprocessor_name --save_dir /home/edlab/jylee/RELIABLE/save_dir/OOD --batch_size 4
    done;
  done;
done