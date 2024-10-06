MODEL_TYPE=bytetrack
MODEL_CONFIG=bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval
DETECTOR_TYPE=yolox
DETECTOR_NAME=yolox_x_8xb8-300e_coco
DATASET_DIR=./data/MOT17_all

for RUN in 1
do
    for SPLIT in train val test
    do
        for EPOCH in {300..300..10}
        do
            echo "$SPLIT run $RUN, epoch $EPOCH"
            python cli.py track \
                --config ./configs/$MODEL_TYPE/$MODEL_CONFIG.py \
                --ckp-detector ./work_dirs/$DETECTOR_TYPE/${DETECTOR_NAME}_mot_all_run$RUN/epoch_$EPOCH.pth \
                --input-dir $DATASET_DIR/$SPLIT/ \
                --output-dir ./detections/$MODEL_TYPE/$SPLIT/run_$RUN/epoch_$EPOCH \
                --device cuda
        done
    done
done