python train.py --method hyper_maml --model Conv4Pool --dataset CUB --num_classes 200 --n_shot 1 --test_n_way 5 --train_n_way 5 --train_aug --stop_epoch 1000 --es_threshold 20 --lr_scheduler multisteplr --lr 1e-3 --milestones 101 1000 --hn_head_len 3 --hm_enhance_embeddings True --hm_update_operator minus --bm_method two_encoders --bm_backbone_weights --bm_activation sigmoid --bm_fixed_size_mask --bm_layer_size 256 --bm_num_layers 3
