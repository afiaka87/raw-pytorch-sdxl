uv run python generate_images.py \
  --lora-checkpoint lora_step_1800_lowest_loss.pt \
  --prompt-file 'dalle-blog-xl.txt' \
  --num-images 1156222 \
  --batch-size 64 \
  --image-size 256 \
  --output-dir './dalle-blog-xl' \
  --shuffle-prompts
