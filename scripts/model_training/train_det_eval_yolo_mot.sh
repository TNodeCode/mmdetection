export DATASET_DIR=data/coco
export CLASSES="classes.txt"
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/instances_train2017.json
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/instances_val2017.json
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/instances_test2017.json
export IMAGES_TRAIN=$DATASET_DIR/train2017
export IMAGES_VAL=$DATASET_DIR/val2017
export IMAGES_TEST=$DATASET_DIR/test2017

export MODEL_TYPE=yolox
export MODEL_NAME=yolox_x_8xb8-300e_coco
export BATCH_SIZE=4
export EPOCHS=300
export WORK_DIR=work_dirs/$MODEL_TYPE/$MODEL_NAME

export DATA_ROOT=data/MOT17_all/
export ANN_TRAIN=train_cocoformat_all.json
export ANN_VAL=val_cocoformat_all.json
export ANN_TEST=test_cocoformat_all.json

OPTIM_SDG="optim_wrapper.optimizer.type=SGD \
    optim_wrapper.optimizer.lr=0.02 \
    optim_wrapper.optimizer.momentum=0.9 \
    optim_wrapper.optimizer.weight_decay=0.0001"

OPTIM_ADAMW="optim_wrapper.optimizer.type=AdamW \
    optim_wrapper.optimizer.lr=5e-3 \
    optim_wrapper.optimizer.weight_decay=0.05"

python tools/train.py \
    configs/$MODEL_TYPE/$MODEL_NAME.py \
    --resume \
    --work-dir $WORK_DIR \
    --cfg-options \
    train_dataset.dataset.data_root=${DATA_ROOT} \
    train_dataset.dataset.ann_file=annotations/$ANN_TRAIN \
    train_dataset.dataset.data_prefix.img=train/ \
    train_dataloader.dataset.dataset.data_root=${DATA_ROOT} \
    train_dataloader.dataset.dataset.ann_file=annotations/$ANN_TRAIN \
    train_dataloader.dataset.dataset.data_prefix.img=train/ \
    val_dataset.dataset.data_root=${DATA_ROOT} \
    val_dataset.dataset.data_prefix.img=val/ \
    val_dataset.dataset.ann_file=annotations/$ANN_VAL \
    val_dataloader.dataset.data_root=${DATA_ROOT} \
    val_dataloader.dataset.data_prefix.img=val/ \
    val_dataloader.dataset.ann_file=annotations/$ANN_VAL \
    val_evaluator.ann_file=${DATA_ROOT}annotations/$ANN_VAL \
    test_dataset.dataset.data_root=${DATA_ROOT} \
    test_dataset.dataset.data_prefix.img=test/ \
    test_dataset.dataset.ann_file=annotations/$ANN_TEST \
    test_dataloader.dataset.data_root=${DATA_ROOT} \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_dataloader.dataset.ann_file=annotations/$ANN_TEST \
    test_evaluator.ann_file=${DATA_ROOT}annotations/$ANN_TEST \
    train_cfg.max_epochs=$EPOCHS \
    $OPTIM_ADAMW \
    default_hooks.logger.interval=10 \
    default_hooks.checkpoint.interval=10 \
    default_hooks.checkpoint.max_keep_ckpts=100 \
    model.bbox_head.num_classes=1 \
    train_dataloader.batch_size=$BATCH_SIZE \
    train_dataloader.num_workers=1 \
    val_dataloader.batch_size=$BATCH_SIZE \
    val_dataloader.num_workers=1 \
    test_dataloader.batch_size=$BATCH_SIZE \
    test_dataloader.num_workers=1 \

# Detect bounding boxes for all datasets and epochs
for DATASET in "train" "val" "test"
do
    for EPOCH in $(seq 1 $EPOCHS)
    do
        python cli.py detect \
            --model-type $MODEL_TYPE \
            --model-name $MODEL_NAME \
            --weight-file epoch_$EPOCH.pth \
            --image-files $DATASET_DIR/$DATASET"'2017/*.png'" \
            --results-file detections_${DATASET}_epoch_${EPOCH}.csv \
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
