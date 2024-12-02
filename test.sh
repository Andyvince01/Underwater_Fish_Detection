#!/bin/bash

#===================================================================
# Optimized Bash Script for Testing with Multiple Models
# Author: Andrea Vincenzo Ricciardi
# Description: This script automates multiple testing runs by 
# looping through different models or batch sizes, logging outputs.
#===================================================================

# ---- Configurable Variables ----
PYTHON_BIN="/bin/python3"
TEST_SCRIPT="test.py"

# ---- Default Parameters ----
BATCH_SIZES=(32)                                                        # Array of batch sizes to loop through
MODELS=(
    #--- Trained Models on the FishScale_Dataset ---#
    # "FishScale_Dataset/yolov8s"
    # "FishScale_Dataset/yolov8s-p2"
    # "FishScale_Dataset/yolov8s-p2-SPD"
    # "FishScale_Dataset/yolov8s-p2-CBAM"
    # "FishScale_Dataset/yolov8s-FishScale"
    # #--- Trained Models on the FishScale + Private_Dataset ---#
    # "FishScale+Private_Dataset/yolov8s"
    # "FishScale+Private_Dataset/yolov8s-p2"
    # "FishScale+Private_Dataset/yolov8s-p2-SPD"
    # "FishScale+Private_Dataset/yolov8s-p2-CBAM"
    # "FishScale+Private_Dataset/yolov8s-FishScale"
    # "FishScale+Private_Dataset/yolov8s-FishScale2"
    # "FishScale+Private_Dataset/yolov8s-FishScale3"
    #--- UIE ---#
    "UIE/yolov8s-FishScale_FunieGAN(freeze)"
    # "UIE/yolov8s-FishScale_FunieGAN"
    # "UIE/yolov8s-FishScale_UIEDM (all)"
    # "UIE/yolov8s-FishScale_UIEDM (freeze)"
)    
MODE_TYPES=('test')           # Default operational mode is 'test'. It can be 'test', 'train' or 'val'

# ---- Parse Command-Line Arguments ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE_TYPES="$2"         # Capture the numerical freeze value
      shift 2                 # Shift past the flag and its value
      ;;
    *)
      MODELS+=("$1")          # Add any non-flag arguments to the model list
      shift
      ;;
  esac
done

# ---- Error Handling ----
trap 'echo ">>> Error occurred during testing. Check terminal output for details." >&2' ERR

# ---- Check Python and Script Existence ----
if ! [ -x "$(command -v $PYTHON_BIN)" ]; then
  echo ">>> Error: Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
  echo ">>> Error: Testing script not found at $TEST_SCRIPT" >&2
  exit 1
fi

# ---- Loop over Models and Batch Sizes ----
for MODEL in "${MODELS[@]}"; do
  for MODE in "${MODE_TYPES[@]}"; do
    # ---- Execute the Training ----
    # $PYTHON_BIN $TEST_SCRIPT --model $MODEL --weights "yolov8s-FishScale3.pt" --kwargs batch=$BATCH_SIZE freeze=$FREEZE name="$MODEL-full" 
    # $PYTHON_BIN $TEST_SCRIPT --model $MODEL --mode $MODE --source "data/fishscale_dataset/train/images/A000065_L.avi.139397.png"
    $PYTHON_BIN $TEST_SCRIPT --model $MODEL --mode $MODE
    # ---- Check for Errors ----
    if [ $? -eq 0 ]; then
      echo ">>> Testing completed successfully for $MODEL with mode=$MODE."
    else
      echo ">>> Testing failed for $MODEL with mode=$MODE." >&2
    fi
  done
done

echo ">>> All training runs completed!"


nohup "" > log.txt &