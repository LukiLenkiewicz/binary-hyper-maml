export NEPTUNE_PROJECT="lukilenkiewicz/hypernets"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOWMyNmQxMi0wYzMxLTQ4ZjgtYmY4YS1mZjNiYzQ1NjVjMzkifQ=="

python train.py --method binary_maml --model Conv4 --dataset cross_char --num_classes 4112 --n_shot 1 --test_n_way 5 --train_n_way 5 --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 --milestones 51 550 --hn_head_len 3 --hm_enhance_embeddings True --hm_update_operator minus --bm_method one_encoder --bm_activation sigmoid --bm_layer_size 512 --bm_num_layers 2 --bm_skip_eval
