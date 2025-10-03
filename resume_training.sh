#!/bin/bash
# Resume training from latest checkpoint

# This script demonstrates how to resume training from the latest checkpoint

# Directory where checkpoints are saved
CHECKPOINT_DIR="./checkpoints/0034"

# Find the latest LoRA checkpoint and inspect it
echo "Finding latest checkpoint in $CHECKPOINT_DIR..."
python resume_training.py --auto "$CHECKPOINT_DIR"

# To automatically execute the resume command, uncomment:
# python resume_training.py --auto "$CHECKPOINT_DIR" --execute

# Or manually resume from a specific checkpoint:
# python train_sdxl.py \
#   --resume_from 'checkpoints/0034/lora_step000500_20241003_162048.pt' \
#   --data_dir '/home/sam/Data/dalle-blog-data/captioned-dalle/' \
#   --use_ema \
#   --use_lora \
#   --lora_rank 16 \
#   --lora_target_mode all \
#   --batch_size 48 \
#   --gradient_accumulation_steps 8 \
#   --num_epochs 1 \
#   --learning_rate 1e-5 \
#   --min_snr_gamma 5.0 \
#   --warmup_steps 500 \
#   --precision bf16 \
#   --image_size 256 \
#   --center_crop \
#   --random_flip \
#   --log_interval 10 \
#   --validation_interval 100 \
#   --num_validation_images 4 \
#   --wandb_project 'raw-sdxl-dalle-blog' \
#   --device 'cuda' \
#   --num_workers 32 \
#   --seed 420