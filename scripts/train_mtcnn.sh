#!/bin/bash
python -m src.fatigue_detection.scripts.train \
  --model_type mtcnn \
  --data_dir data/raw \
  --batch_size 32 \
  --epochs 50 \
  --output_dir models 