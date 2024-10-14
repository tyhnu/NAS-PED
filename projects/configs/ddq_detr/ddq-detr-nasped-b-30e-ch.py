_base_ = [
    '../_base_/datasets/crowdhuman_detection.py', '../_base_/default_runtime.py'
]
custom_imports = dict(imports=['projects'], allow_failed_imports=False)

pretrained = '/home/lbx/lbx2/DDQ-ddq_detr/checkpoint_l.pth'
model = dict(
    type='DDQDETR',
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(type='DetDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=1),
    backbone=dict(
        type='SearchedIBLarge',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(type='ChannelMapper',
              in_channels=[64, 192, 320, 448],
              kernel_size=1,
              out_channels=256,
              act_cfg=None,
              norm_cfg=dict(type='GN', num_groups=32),
              num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(type='DDQDETRHead',
                   num_classes=1,
                   sync_cls_avg_factor=True,
                   loss_cls=dict(type='FocalLoss',
                                 use_sigmoid=True,
                                 gamma=2.0,
                                 alpha=0.25,
                                 loss_weight=1.0),
                   loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                   loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(label_noise_scale=0.5,
                box_noise_scale=1.0,
                group_cfg=dict(dynamic=True,
                               num_groups=None,
                               num_dn_queries=100)),
    train_cfg=dict(assigner=dict(
        type='HungarianAssigner',
        match_costs=[
            dict(type='FocalLossCost', weight=2.0),
            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])),
    test_cfg=dict(max_per_img=500))

train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(type='RandomChoiceResize',
                     scales=[
                         (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                         (736, 1333), (768, 1333), (800, 1333)
                     ],
                     keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(type='RandomCrop',
                     crop_type='absolute_range',
                     crop_size=(384, 600),
                     allow_negative_crop=True),
                dict(type='RandomChoiceResize',
                     scales=[
                         (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                         (736, 1333), (768, 1333), (800, 1333)
                     ],
                     keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(filter_cfg=dict(filter_empty_gt=False),
                                     pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.000025, weight_decay=0.05),
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))

max_epochs = 30
train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=max_epochs,
                 val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.0001,
         by_epoch=False,
         begin=0,
         end=2000),
    dict(type='MultiStepLR',
         begin=0,
         end=max_epochs,
         by_epoch=True,
         milestones=[20, 26],
         gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

# base_batch_size = (8 GPUs) x (2 samples per GPU)
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
#
# # file_client_args = dict(
# #     backend='petrel',
# #     path_mapping=dict({
# #         './data/': 's3://openmmlab/datasets/detection/',
# #         'data/': 's3://openmmlab/datasets/detection/'
# #     }))
# file_client_args = dict(backend='disk')
# auto_scale_lr = dict(base_batch_size=16)
# test_dataloader = dict(batch_size=1,
#                        num_workers=2,
#                        persistent_workers=True,
#                        drop_last=False,
#                        sampler=dict(type='DefaultSampler', shuffle=False),
#                        dataset=dict(
#                            type=dataset_type,
#                            data_root=data_root,
#                            ann_file='annotations/image_info_test-dev2017.json',
#                            data_prefix=dict(img='test2017/'),
#                            test_mode=True,
#                        ))
# test_evaluator = dict(type='CocoMetric',
#                       metric='bbox',
#                       format_only=True,
#                       ann_file=data_root +
#                       'annotations/image_info_test-dev2017.json',
#                       outfile_prefix='./work_dirs/coco_detection/test')
