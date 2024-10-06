python ./tools/test_tracking.py \
    ./configs/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py \
    --detector ./work_dirs/yolox/yolox_x_8xb8-300e_coco_run2/epoch_300.pth \
    --reid ./work_dirs/reid_r50_8xb32-6e_mot17train80_test-mot17val20/epoch_6.pth \