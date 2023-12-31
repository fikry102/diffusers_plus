export MODEL_NAME="../../stable-diffusion-v1-4"
export DATASET_NAME="../../pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_xformers_memory_efficient_attention \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" 

  # --max_train_steps=15000 \