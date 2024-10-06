export CONFIG_DIR=projects/CO-DETR/configs

export DATASET_DIR=data/MOT17_all/
export IMAGES_TRAIN=$DATASET_DIR/train
export IMAGES_VAL=$DATASET_DIR/val
export IMAGES_TEST=$DATASET_DIR/test

export MODEL_TYPE=codino
export MODEL_NAME=co_dino_5scale_r50_lsj_8xb2_1x_coco
export BATCH_SIZE=2
export EPOCHS=25
export WORK_DIR=work_dirs/$MODEL_TYPE/$MODEL_NAME

export ANN_TRAIN=train_cocoformat_all.json
export ANN_VAL=val_cocoformat_all.json
export ANN_TEST=test_cocoformat_all.json
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/$ANN_TRAIN
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/$ANN_VAL
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/$ANN_TEST


OPTIM_SDG="optim_wrapper.optimizer.type=SGD \
    optim_wrapper.optimizer.lr=0.02 \
    optim_wrapper.optimizer.momentum=0.9 \
    optim_wrapper.optimizer.weight_decay=0.0001"

OPTIM_ADAMW="optim_wrapper.optimizer.type=AdamW \
    optim_wrapper.optimizer.lr=1e-4 \
    optim_wrapper.optimizer.weight_decay=0.05"


python tools/train.py \
    $CONFIG_DIR/$MODEL_TYPE/$MODEL_NAME.py \
    --work-dir work_dirs/$MODEL_TYPE/$MODEL_NAME \
    --cfg-options \
    train_dataloader.dataset.dataset.data_root=${DATASET_DIR} \
    train_dataloader.dataset.dataset.ann_file=annotations/$ANN_TRAIN \
    train_dataloader.dataset.dataset.data_prefix.img=train/ \
    val_dataloader.dataset.data_root=${DATASET_DIR} \
    val_dataloader.dataset.data_prefix.img=val/ \
    val_dataloader.dataset.ann_file=annotations/$ANN_VAL \
    val_evaluator.ann_file=${DATASET_DIR}annotations/$ANN_VAL \
    test_dataloader.dataset.data_root=${DATASET_DIR} \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_dataloader.dataset.ann_file=annotations/$ANN_TEST \
    test_evaluator.ann_file=${DATASET_DIR}annotations/$ANN_TEST \
    train_cfg.max_epochs=$EPOCHS \
    default_hooks.logger.interval=10 \
    model.query_head.num_classes=1 \
    model.roi_head.0.bbox_head.num_classes=1 \
    model.bbox_head.0.num_classes=1 \
    train_dataloader.batch_size=$BATCH_SIZE \
    val_dataloader.batch_size=$BATCH_SIZE \
    test_dataloader.batch_size=$BATCH_SIZE

exit
# Detect bounding boxes for all datasets and epochs
for SPLIT in "train" "val" "test"
do
    for EPOCH in $(seq 1 $EPOCHS)
    do
        python cli.py detect \
            --model-type $MODEL_TYPE \
            --model-name $MODEL_NAME \
            --work-dir work_dirs/$MODEL_TYPE/$MODEL_NAME \
            --dataset-dir ${DATASET_DIR}${SPLIT}/ \
            --weight-file epoch_$EPOCH.pth \
            --image-files "'**/img/*.png'" \
            --results-file detections_${SPLIT}_epoch_${EPOCH}.csv \
            --batch-size $BATCH_SIZE \
            --score-threshold 0.0 \
            --device cuda:0
    done
done

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
