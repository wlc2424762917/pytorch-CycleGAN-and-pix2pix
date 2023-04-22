
CUDA_VISIBLE_DEVICES=1 nohup python train_m.py --name LGE_DTI_new_semloss --model cycle_gan --display_id -1 --batch_size 16 >>log_cycle_LGE_DTI_new_sem_loss &