# dataset settings
dataset_type = 'CityPersonDataset'
data_root = '/home/lbx/lbx2/DATA/cityperson/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(batch_size=1,
                        num_workers=2,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=dict(
                            type=dataset_type,
                            data_root=data_root,
                            ann_file=data_root + 'train_cp_new_mmdet.json',
                            # ann_file=data_root + 'train_resonable.json',
                            data_prefix=dict(img=''),
                            filter_cfg=dict(filter_empty_gt=True, min_size=1),
                            pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(
                          type=dataset_type,
                          data_root=data_root,
                          ann_file=data_root + 'val_gt_for_mmdetction.json',
                          # ann_file=data_root + 'valdebug.json',
                          data_prefix=dict(img='/home/lbx/lbx2/DATA/cityperson//leftImg8bit_trainvaltest/leftImg8bit/val/'),
                          test_mode=True,
                          pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CityPersonMetric',
                     ann_file=data_root + 'val_gt_for_mmdetction.json',
                     metric=['MR'],
                     format_only=False)
test_evaluator = val_evaluator

