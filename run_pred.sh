python predict.py \
 --data_dir='./data/huangshan' \
 --data_set='huangshan' \
 --config_json='./configs/config.json' \
 --pretrained_ckpt='./ckpts/huangshan/train' \
 --base_directory='./ckpts/huangshan' \
 --mode='TEST' \
 --optimizer='adam' \
 --max_steps=1200 \
 --summaries_every=10 \
 --print_every=10 \
 --max_num_to_print=3 \
 --attention_option='luong' \
