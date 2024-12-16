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



accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 sd3_train_network.py \
--pretrained_model_name_or_path ./sd3/stabilityai-stable-diffusion-3.5-large/sd3.5_large.safetensors \
--clip_l ./sd3/clip_l.safetensors --clip_g ./sd3/clip_g.safetensors \
--t5xxl ./sd3/t5xxl_fp8_e4m3fn.safetensors --cache_latents_to_disk --save_model_as safetensors --sdpa \
--persistent_data_loader_workers --max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing \
--mixed_precision bf16 --save_precision bf16 --network_module networks.lora_sd3 \
--network_dim 16 --network_args "loraplus_unet_lr_ratio=4" --network_train_unet_only \
--optimizer_type adamw8bit --learning_rate 1e-3 --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
--fp8_base --highvram --max_train_epochs 8 --save_every_n_epochs 1 \
--dataset_config config.toml --output_dir lora \
--output_name sd3-ori1ori2-lora --sample_prompts=prompts_ori12.txt \
--sample_every_n_epochs 1 --sample_at_first