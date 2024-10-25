#!/bin/bash

#===================================================================
# Optimized Bash Script for Training with Multiple Configurations
# Author: Andrea Vincenzo Ricciardi
# Description: This script automates multiple training runs by 
# looping through different models or batch sizes, logging outputs.
#===================================================================

# ---- Configurable Variables ----
PYTHON_BIN="/bin/python3"
TRAIN_SCRIPT="train.py"
FUSE_SCRIPT="models.misc"

# ---- Default Parameters ----
BATCH_SIZES=(32)                                                        # Array of batch sizes to loop through
MODELS=(
    # "yolov8s"
    # "yolov8s-p2"
    # "yolov8s-p2-SPD"
    # "yolov8s-p2-CBAM"
    "yolov8s-FishScale"
    # "yolov8s-FishScale_FunieGAN"
)    

WEIGHTS_VALUES=('yolov8s-FishScale-full.pt')
FREEZE_VALUES=(0)  # Default freeze value is 0 (no freezing)

# ---- Parse Command-Line Arguments ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --freeze)
      FREEZE_VALUES="$2"      # Capture the numerical freeze value
      shift 2                 # Shift past the flag and its value
      ;;
    *)
      MODELS+=("$1")          # Add any non-flag arguments to the model list
      shift
      ;;
  esac
done

# ---- Error Handling ----
trap 'echo "Error occurred during training. Check terminal output for details." >&2' ERR

# ---- Check Python and Script Existence ----
if ! [ -x "$(command -v $PYTHON_BIN)" ]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Error: Training script not found at $TRAIN_SCRIPT" >&2
  exit 1
fi

# ---- Loop over Models and Batch Sizes ----
for MODEL in "${MODELS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for FREEZE in "${FREEZE_VALUES[@]}"; do
      for WEIGHTS in "${WEIGHTS_VALUES[@]}"; do

      # Check if weights is empty
      if [ -z "$WEIGHTS" ]; then
        echo "Training $MODEL with batch size $BATCH_SIZE and freeze $FREEZE (weights = $WEIGHTS)..."
        $PYTHON_BIN $TRAIN_SCRIPT --model $MODEL --kwargs batch=$BATCH_SIZE freeze=$FREEZE name="$MODEL-full" 
      else
        echo "Training $MODEL with batch size $BATCH_SIZE and freeze $FREEZE with weights $WEIGHTS..."
        $PYTHON_BIN $TRAIN_SCRIPT --model $MODEL --weights $WEIGHTS --kwargs batch=$BATCH_SIZE freeze=$FREEZE name="$MODEL-full" 
      fi

      done
   
      # # ---- Set Freeze Name ----
      # if [[ "$FREEZE" =~ ^[0-40]$ ]]; then  # Check if FREEZE is either 0 or 1
      #         FREEZE_NAME=$( [ "$FREEZE" -gt 0 ] && echo "freeze layer from 0 to $FREEZE" || echo "no_freeze" )
      # else
      #   echo "Error: FREEZE value '$FREEZE' is not a valid integer (0 or 40)."
      #   exit 1
      # fi

      # echo "Training $MODEL with batch size $BATCH_SIZE and freeze $FREEZE_NAME..."

      # ---- Check for Errors ----
      if [ $? -eq 0 ]; then
        echo "Training completed successfully for $MODEL with batch size $BATCH_SIZE and freeze $FREEZE."
      else
        echo "Training failed for $MODEL with batch size $BATCH_SIZE and freeze $FREEZE." >&2
      fi
    done
  done
done

echo "All training runs completed."