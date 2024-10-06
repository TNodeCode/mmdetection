export DATA_ROOT=data/MOT17_all/
export ANN_TRAIN=train_cocoformat_all.json
export ANN_VAL=val_cocoformat_all.json
export ANN_TEST=test_cocoformat_all.json
export NUM_OBJECTS=1000
export MARGIN=0.3

python ./tools/test_tracking.py \
    ./configs/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --detector ./work_dirs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco_mot_all_adamw_run3/epoch_9.pth \
    --reid ./work_dirs/reid_r50_8xb32-6e_mot17train80_test-mot17val20/epoch_6.pth \
    --cfg-options \
    model.reid.head.num_classes=${NUM_OBJECTS} \
    model.reid.head.loss_triplet.margin=${MARGIN} \
    model.tracker.match_iou_thr=0.3 \
    val_dataloader.dataset.data_root=${DATA_ROOT} \
    val_dataloader.dataset.data_prefix.img_path=val/ \
    val_dataloader.dataset.ann_file=annotations/$ANN_VAL \
    test_dataloader.dataset.data_root=${DATA_ROOT} \
    test_dataloader.dataset.data_prefix.img_path=test/ \
    test_dataloader.dataset.ann_file=annotations/$ANN_TEST \
