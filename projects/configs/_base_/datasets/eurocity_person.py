# dataset settings
dataset_type = 'CocoPersonDataset'
data_root = '/data2/lbx/DATA/ECP/'

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1920, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1920, 1024), keep_ratio=True),
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
                            ann_file='/data2/lbx/DDQ-ddq_detr/projects/configs/_base_/datasets/EuroCity/day_train_all_corrected.json',
                            data_prefix=dict(img='/data2/lbx/DATA/'),
                            filter_cfg=dict(filter_empty_gt=True, min_size=32),
                            pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(
                          type=dataset_type,
                          data_root=data_root,
                          ann_file='/data2/lbx/DDQ-ddq_detr/projects/configs/_base_/datasets/EuroCity/day_val.json',
                          data_prefix=dict(img='/data2/lbx/DATA/'),
                          test_mode=True,
                          pipeline=test_pipeline))
# test_dataloader = val_dataloader
test_dataloader = dict(batch_size=1,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(
                          type=dataset_type,
                          data_root=data_root,
                          ann_file='/data2/lbx/DDQ-ddq_detr/projects/configs/_base_/datasets/EuroCity/day_val.json',
                          data_prefix=dict(img='/data2/lbx/DATA/'),
                          test_mode=True,
                          pipeline=test_pipeline))
# test_dataloader = val_dataloader
val_evaluator = dict(type='EuroCityPerson',
                     ann_file='/data2/lbx/DDQ-ddq_detr/projects/configs/_base_/datasets/EuroCity/day_val.json',
                     metric='MR',
                     format_only=False)
test_evaluator = dict(type='EuroCityPerson',
                     ann_file='/data2/lbx/DDQ-ddq_detr/projects/configs/_base_/datasets/EuroCity/day_val.json',
                     metric='MR',
                     format_only=False)
