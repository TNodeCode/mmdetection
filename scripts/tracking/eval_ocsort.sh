MODEL_TYPE=ocsort
MODEL_CONFIG=ocsort_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval
DETECTOR_TYPE=yolox
DETECTOR_NAME=yolox_x_8xb8-300e_coco

for RUN in 2
do
    for SPLIT in test
    do
        for EPOCH in {10..300..10} 
        do
            echo "$SPLIT run $RUN, epoch $EPOCH"
            python cli.py track \
                --config ./configs/$MODEL_TYPE/$MODEL_CONFIG.py \
                --ckp-detector ./work_dirs/$DETECTOR_TYPE/${DETECTOR_NAME}_run$RUN/epoch_$EPOCH.pth \
                --input-dir ./data/MOT17/$SPLIT/ \
                --output-dir ./detections/$MODEL_TYPE/$SPLIT/run_$RUN/epoch_$EPOCH \
                --device cuda
        done
    done
done