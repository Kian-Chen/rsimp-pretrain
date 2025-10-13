model_name=MAE
#32 64 128 256
for batch_size in 16 #32 64 128 256
do
for lr in 0.001
do
for mask_rate in 0.2 0.5 0.7 0.9
do


python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --data_path ./EuroSAT_MS/ \
  --log_dir ./logs/${model_name}/mask${mask_rate}/ \
  --log_name log.txt \
  --model_id mask${mask_rate} \
  --mask_rate ${mask_rate} \
  --model $model_name \
  --data eurosat_ms \
  --d_model 192 \
  --c_in 13 \
  --c_out 13 \
  --patience 4 \
  --batch_size $batch_size \
  --train_epoch 100 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --gpu 5

done
done
done