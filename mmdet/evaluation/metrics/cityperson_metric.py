# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os.path as osp
import tempfile
from collections import OrderedDict
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient, dump, load
from mmengine.logging import MMLogger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.registry import METRICS

from ..CityPerson.coco import COCO
from ..CityPerson.eval_MR_multisetup import COCOeval

PERSON_CLASSES = ['background', 'person']


@METRICS.register_module()
class CityPersonMetric(BaseMetric):
    """CrowdHuman evaluation metric.

    Evaluate Average Precision (AP), Miss Rate (MR) and Jaccard Index (JI)
    for detection tasks.

    Args:
        ann_file (str): Path to the annotation file.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'AP', 'MR' and 'JI'. Defaults to 'AP'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        eval_mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
        iou_thres (float): IoU threshold. Defaults to 0.5.
        compare_matching_method (str, optional): Matching method to compare
            the detection results with the ground_truth when compute 'AP'
            and 'MR'.Valid method include VOC and None(CALTECH). Default to
            None.
        mr_ref (str): Different parameter selection to calculate MR. Valid
            ref include CALTECH_-2 and CALTECH_-4. Defaults to CALTECH_-2.
        num_ji_process (int): The number of processes to evaluation JI.
            Defaults to 10.
    """
    default_prefix: Optional[str] = 'cityperson'

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = ['MR'],
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        # crowdhuman evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MR']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric should be one of 'MR', 'AP', 'JI',"
                               f'but got {metric}.')

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        self.outfile_prefix = outfile_prefix
        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)

        if ann_file is not None:
            with self.file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None
        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None
    # @staticmethod

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = 1
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = 1
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

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
        # assert isinstance (results, list), 'results must be a list'
        # assert len (results) == len (self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #         format (len (results), len (self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory ()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        res = []
        for idx, result in enumerate (results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = 1
                res.append(data)
        jsonfile_prefix = f'{jsonfile_prefix}.bbox.json'
        with open(jsonfile_prefix, 'w') as f:
            json.dump(res, f)
        return jsonfile_prefix, tmp_dir

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            eval_results(Dict[str, float]): The computed metrics.
            The keys are the names of the metrics, and the values
            are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # print(results)
        gts, preds = zip(*results)
        tmp_dir = None
        outfile_prefix=None

        # convert predictions to coco format and dump to json file
        result_file,tmp_dir = self.format_results(preds, outfile_prefix)
        # result_file = self.results2json(preds, outfile_prefix)

        mean_MR = []
        my_id_setup = []
        # eval_results = {}
        # cocoGt = self.coco
        for id_setup in range (0, 4):
            cocoGt = COCO(self.ann_file)
            # cocoGt = self.coco
            cocoDt = cocoGt.loadRes(result_file)
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

        return eval_results

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()


            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    @staticmethod
    def gather(results):
        """Integrate test results."""
        assert len(results)
        img_num = 0
        for result in results:
            if result['n'] != 0 or result['m'] != 0:
                img_num += 1
        mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
        valids = np.sum([rb['k'] for rb in results])
        total = np.sum([rb['n'] for rb in results])
        gtn = np.sum([rb['m'] for rb in results])
        line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}'\
            .format(mean_ratio, valids, total, gtn)
        return line, mean_ratio


    def get_ignores(self, dt_boxes, gt_boxes):
        """Get the number of ignore bboxes."""
        if gt_boxes.size:
            ioas = bbox_overlaps(dt_boxes, gt_boxes, mode='iof')
            ioas = np.max(ioas, axis=1)
            rows = np.where(ioas > self.iou_thres)[0]
            return len(rows)
        else:
            return 0



