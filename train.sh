# python train.py --method hyper_maml --model Conv4 --dataset cross_char --num_classes 4112 --n_shot 1 --test_n_way 5 --train_n_way 5 --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 --milestones 51 550 --hn_head_len 3 --hm_enhance_embeddings True --hm_update_operator minus --bm_method one_encoder --bm_activation sigmoid --bm_layer_size 512 --bm_num_layers 2 --bm_skip_eval
python train.py --method hyper_maml --model Conv4 --dataset cross_char --num_classes 4112 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 \
  --hm_maml_warmup --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 --milestones 51 550 \
  --hn_head_len 3 --hn_hidden_size 512 --hm_enhance_embeddings True --hm_use_class_batch_input