export MODEL_NAME="../../stable-diffusion-v1-4"
export DATASET_NAME="../../pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" --multi_gpu train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora" \
  --validation_prompt="cute dragon creature"

#--train_batch_size=1 \
#-num_train_epochs=100 --checkpointing_steps=5000 \