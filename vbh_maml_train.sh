# python train.py --method vbh_maml --dataset cross_char --num_classes 4112 --n_shot 1 --test_n_way 5 \
#     --train_n_way 5 --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-2 --milestones 51 550 --hn_head_len 3 \
#     --hm_enhance_embeddings True --hm_update_operator minus --bm_layer_size 512 --bm_num_layers 2 \
#     --bm_mask_size 0.2 --bm_chunk_size 310 --hn_tn_depth 4 --hn_tn_hidden_size 512

python train_vbh.py --method vbh_maml --dataset cross_char --num_classes 4112 \
  --n_shot 1 --test_n_way 5 --train_n_way 5 \
  --stop_epoch 64 --lr_scheduler multisteplr --lr 1e-3 \
  --milestones 51 550 \
  --hn_head_len 2 --hn_hidden_size 256 --hm_enhance_embeddings True \
  --hn_tn_depth 2 --hn_tn_hidden_size 256 --hn_use_mask