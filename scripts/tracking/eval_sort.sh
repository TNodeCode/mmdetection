MODEL_TYPE=sort
MODEL_CONFIG=sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval
DETECTOR_TYPE=faster_rcnn
DETECTOR_NAME=faster-rcnn_r50_fpn_1x_coco_mot_all_adamw
DATASET_DIR=./data/MOT17_all

for RUN in 4
do
    for SPLIT in train val test
    do
        for EPOCH in 7
        do
            echo "$SPLIT run $RUN, epoch $EPOCH"
            python cli.py track \
                --config ./configs/$MODEL_TYPE/$MODEL_CONFIG.py \
                --ckp-detector ./work_dirs/$DETECTOR_TYPE/${DETECTOR_NAME}_run$RUN/epoch_$EPOCH.pth \
                --input-dir $DATASET_DIR/$SPLIT/ \
                --output-dir ./detections/$MODEL_TYPE/$SPLIT/run_$RUN/epoch_$EPOCH \
                --device cuda
        done
    done
done