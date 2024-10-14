import copy
import json
import os.path as osp
import tempfile
import mmcv
from typing import List, Union

import numpy as np

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

from .api_wrappers import COCO


@DATASETS.register_module()
class CityPersonDataset(BaseDetDataset):

    METAINFO = {
        'classes':  ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
             (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
             (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
             (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
             (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
             (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
             (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
             (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
             (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
             (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
             (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
             (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
             (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
             (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
             (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
             (246, 0, 122), (191, 162, 208)]
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            # if ann['area'] <= 0 or w < 1 or h < 1:
            #     continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        # print(ann_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos


    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set ()
        for i, class_id in enumerate (self.cat_ids):
            ids |= set (self.coco.catToImgs[class_id])
        self.img_ids = list (ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs ([i])[0]
            info['filename'] = info['file_name']
            data_infos.append (info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate (ann_info):
            if ann.get ('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get ('iscrowd', False):
                gt_bboxes_ignore.append (bbox)
            else:
                gt_bboxes.append (bbox)
                gt_labels.append (self.cat2label[ann['category_id']])
                if "segmentation" not in ann:
                    bbox = ann['bbox']
                    ann['segmentation'] = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                                            bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1] + bbox[3]]]
                gt_masks_ann.append (ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array (gt_bboxes, dtype=np.float32)
            gt_labels = np.array (gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros ((0, 4), dtype=np.float32)
            gt_labels = np.array ([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array (gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros ((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace ('jpg', 'png')

        ann = dict (
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist ()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range (len (self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range (bboxes.shape[0]):
                data = dict ()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh (bboxes[i])
                data['score'] = float (bboxes[i][4])
                data['category_id'] = 1
                json_results.append (data)
        return json_results

    def _det2json(self, results):
        json_results = []
        for idx in range (len (self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range (len (result)):
                # print(label)
                bboxes = result[label]
                for i in range (bboxes.shape[0]):
                    data = dict ()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh (bboxes[i])
                    data['score'] = float (bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    # data['category_id'] = 1
                    # data['category_id']=0
                    json_results.append (data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range (len (self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range (len (det)):
                # bbox results
                bboxes = det[label]
                for i in range (bboxes.shape[0]):
                    data = dict ()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh (bboxes[i])
                    data['score'] = float (bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append (data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance (seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range (bboxes.shape[0]):
                    data = dict ()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh (bboxes[i])
                    data['score'] = float (mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance (segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode ()
                    data['segmentation'] = segms[i]
                    segm_json_results.append (data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict ()
        if isinstance (results[0], list):
            json_results = self._det2json (results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump (json_results, result_files['bbox'])
        elif isinstance (results[0], tuple):
            json_results = self._segm2json (results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump (json_results[0], result_files['bbox'])
            mmcv.dump (json_results[1], result_files['segm'])
        elif isinstance (results[0], np.ndarray):
            json_results = self._proposal2json (results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump (json_results, result_files['proposal'])
        else:
            raise TypeError ('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance (results, list), 'results must be a list'
        assert len (results) == len (self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format (len (results), len (self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory ()
            jsonfile_prefix = osp.join (tmp_dir.name, 'results')
        else:
            tmp_dir = None
        res = []
        for id, boxes in enumerate (results):
            boxes = boxes[0]
            if type (boxes) == list:
                boxes = boxes[0]
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            if len (boxes) > 0:
                for box in boxes:
                    # box[:4] = box[:4] / 0.6
                    temp = dict()
                    temp['image_id'] = id + 1
                    temp['category_id'] = 1
                    temp['bbox'] = box[:4].tolist ()
                    temp['score'] = float (box[4])
                    res.append (temp)
        jsonfile_prefix = f'{jsonfile_prefix}.bbox.json'
        with open(jsonfile_prefix, 'w') as f:
            json.dump(res, f)
        return jsonfile_prefix, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange (0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance (metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError (f'metric {metric} is not supported')
        # print(results)
        result_files, tmp_dir = self.format_results (results, jsonfile_prefix)
        mean_MR = []
        my_id_setup = []
        # eval_results = {}
        # cocoGt = self.coco
        for id_setup in range (0, 4):
            cocoGt = COCO(self.ann_file)
            # cocoGt = self.coco
            cocoDt = cocoGt.loadRes(result_files)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            # cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate(id_setup)
            cocoEval.accumulate()
            mean_MR.append(cocoEval.summarize_nofile(id_setup))
            my_id_setup.append(id_setup)
        eval_results = {
            'Reasonable': mean_MR[0],
            'Reasonable_small': mean_MR[1],
            'Reasonable_occ=heavy': mean_MR[2],
            'all': mean_MR[3]
        }
        # print_log (str (eval_results), logger)
        return eval_results