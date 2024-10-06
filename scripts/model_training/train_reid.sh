export DATA_ROOT=data/MOT17_all
export MARGIN=12.0


python tools/train.py \
    configs/reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py \
    --resume \
    --work-dir work_dirs/reid/reid_mot_all \
    --cfg-options \
    train_cfg.max_epochs=25 \
    model.head.out_channels=256 \
    model.head.num_classes=1000 \
    model.head.loss_triplet.margin=${MARGIN} \
    train_dataloader.dataset.data_root=${DATA_ROOT} \
    val_dataloader.dataset.data_root=${DATA_ROOT} \
    test_dataloader.dataset.data_root=${DATA_ROOT} \
