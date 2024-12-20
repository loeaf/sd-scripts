python train_network.py \
  --pretrained_model_name_or_path="./sd3/Anything-v4.5-pruned.safetensors" \
  --dataset_config="config.toml" \
  --network_module=networks.lora \
  --output_dir="output" \
  --output_name="my_lora_models" \
  --save_every_n_epochs=1 \
  --noise_offset=0.3 \
  --network_alpha=256 \
  --clip_skip=2 \
  --max_train_epochs=10 \
  --learning_rate=1e-4 \
  --text_encoder_lr=1e-4 \
  --unet_lr=1e-4 \
  --lr_scheduler=cosine_with_restarts \
  --lr_warmup_steps=100

#아래 코드로 마지막 학습 진행.
python ./train_network.py --pretrained_model_name_or_path="./sd3/Anything-v4.5-pruned.safetensors" \
--dataset_config=./config.toml \
--network_module=networks.lora \
--save_every_n_epochs=5 \
--output_dir="./my_lora_models/" --output_name=testmac \
--noise_offset=0.1 --optimizer_type=Lion \
--clip_skip=2

CUDA_VISIBLE_DEVICES=1 python ./train_network.py --pretrained_model_name_or_path="./sd3/Anything-v4.5-pruned.safetensors" --dataset_config=./config.toml --network_module=networks.lora --network_dim=128 --network_alpha=64 --save_every_n_epochs=5 --output_dir="./my_lora_models/" --output_name=testmac --noise_offset=0.1 --optimizer_type=Lion --clip_skip=2 --learning_rate=1e-4 --max_train_epochs=50 --lr_scheduler=cosine_with_restarts --lr_warmup_steps=100 --save_state_on_train_end --save_precision=fp16 --mixed_precision=fp16 --noise_offset_random_strength --huber_schedule=snr