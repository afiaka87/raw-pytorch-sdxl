#  Lower Impact Optimizations ✅
#
#  1. pin_memory=True - Already enabled in both DataLoader implementations
#  2. set_to_none=True - Added to all optimizer.zero_grad() calls (3 locations)
#  3. torch.cuda.empty_cache() - Added every 100 steps to prevent fragmentation
#
#  Quantization ✅
#
#  1. --quantize flag - Accepts int8 or 4bit
#  2. INT8 quantization - Uses bitsandbytes.nn.Linear8bitLt
#  3. 4-bit quantization - Uses bitsandbytes.nn.Linear4bit with NF4 format
#  4. Proper ordering - Quantization → LoRA → Gradient Checkpointing
#  5. Memory reporting - Shows before/after quantization footprint

uv run python train_sdxl.py \
	--data_dir '/home/sam/Data/captioned-birds-wds/*.tar' \
	--use_wds \
	--wds_caption_key 'txt' \
	--wds_image_key 'png' \
	--images_only \
        --use_lora \
        --lora_rank 16 \
        --lora_target_mode all \
	--quantize "int8" \
	--use_flash_attention \
	--8_bit_adam \
	--batch_size 16 \
	--gradient_accumulation_steps 4 \
	--num_epochs 1 \
	--learning_rate 1e-6 \
	--min_snr_gamma 5.0 \
	--max_loss_value 0.5 \
	--warmup_steps 2000 \
	--max_grad_norm 0.3 \
	--precision bf16 \
	--image_size 256 \
	--center_crop \
	--random_flip \
	--log_interval=10 \
	--validation_interval 100 \
	--validation_caption_file 'examples/pixelart-256.txt' \
        --validation_guidance_scale 7.5 \
	--num_validation_images 16 \
	--wandb_project 'raw-sdxl-dalle-blog' \
	--device 'cuda' \
	--num_workers 12 \
	--seed 420 \
	--save_interval 500
