export CONFIG_DIR=configs

export DATASET_DIR=data/MOT17_all/
export IMAGES_TRAIN=$DATASET_DIR/train
export IMAGES_VAL=$DATASET_DIR/val
export IMAGES_TEST=$DATASET_DIR/test

export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster-rcnn_r50_fpn_1x_coco_mot_all_adamw_run3
export BATCH_SIZE=2
export EPOCHS=12
export WORK_DIR=work_dirs/$MODEL_TYPE/$MODEL_NAME

export ANN_TRAIN=train_cocoformat_all.json
export ANN_VAL=val_cocoformat_all.json
export ANN_TEST=test_cocoformat_all.json
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/$ANN_TRAIN
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/$ANN_VAL
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/$ANN_TEST

# Evaluate detected bounding boxes for training dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TRAIN \
    --epochs $EPOCHS \
    --score-threshold 0.5 \
    --csv_file_pattern detections_train_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_train.csv

# Evaluate detected bounding boxes for validation dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_VAL \
    --epochs $EPOCHS \
    --score-threshold 0.5 \
    --csv_file_pattern detections_val_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_val.csv

# Evaluate detected bounding boxes for test dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TEST \
    --score-threshold 0.5 \
    --epochs $EPOCHS \
    --csv_file_pattern detections_test_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_test.csv
