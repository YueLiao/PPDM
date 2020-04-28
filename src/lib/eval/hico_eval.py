import json
import numpy as np

class hico():
    def __init__(self, annotation_file):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.train_annotations = json.load(open(annotation_file.replace('test_hico.json', 'trainval_hico.json'),'r'))
        self.overlap_iou = 0.5
        self.verb_name_dict = []
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        self.file_name = []
        self.train_sum = {}
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            gt_bbox = gt_i['annotations']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                triplet = [gt_bbox[gt_hoi_i['subject_id']]['category_id'],gt_bbox[gt_hoi_i['object_id']]['category_id'],gt_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    self.verb_name_dict.append(triplet)
                if self.verb_name_dict.index(triplet) not in self.sum_gt.keys():
                    self.sum_gt[self.verb_name_dict.index(triplet)] =0
                self.sum_gt[self.verb_name_dict.index(triplet)] += 1
        for train_i in self.train_annotations:
            train_hoi = train_i['hoi_annotation']
            train_bbox = train_i['annotations']
            for train_hoi_i in train_hoi:
                if isinstance(train_hoi_i['category_id'], str):
                    train_hoi_i['category_id'] = int(train_hoi_i['category_id'].replace('\n', ''))
                triplet = [train_bbox[train_hoi_i['subject_id']]['category_id'],train_bbox[train_hoi_i['object_id']]['category_id'],train_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                if self.verb_name_dict.index(triplet) not in self.train_sum.keys():
                    self.train_sum[self.verb_name_dict.index(triplet)] =0
                self.train_sum[self.verb_name_dict.index(triplet)] += 1
        for i in range(len(self.verb_name_dict)):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
        self.r_inds = []
        self.c_inds = []
        for id in self.train_sum.keys():
            if self.train_sum[id] < 10:
                self.r_inds.append(id)
            else:
                self.c_inds.append(id)

        self.num_class = len(self.verb_name_dict)
    def evalution(self, predict_annot):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if len(gt_bbox)!=0:
                pred_bbox = self.add_One(pred_i['predictions']) #convert zero-based to one-based indices
                bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                pred_hoi = pred_i['hoi_prediction']
                gt_hoi = gt_i['hoi_annotation']
                self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs, pred_bbox,bbox_ov)
            else:
                pred_bbox = self.add_One(pred_i['predictions']) #convert zero-based to one-based indices
                for i, pred_hoi_i in enumerate(pred_i['hoi_prediction']):
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                               pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_hoi_i['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        for i in range(len(self.verb_name_dict)):
            sum_gt = self.sum_gt[i]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec,prec)
            max_recall[i] = np.max(rec)
            #print('class {} --- ap: {}   max recall: {}'.format(
             #    self.verb_name_dict[i], ap[i-1], max_recall[i-1]))
        mAP = np.mean(ap[:])
        mAP_rare = np.mean(ap[self.r_inds])
        mAP_nonrare = np.mean(ap[self.c_inds])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP nonrare: {}  max recall: {}'.format(mAP, mAP_rare, mAP_nonrare, m_rec))
        print('--------------------')
        return mAP

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs, pred_bbox,bbox_ov):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_obj_ov=bbox_ov[pred_hoi_i['object_id']]
                    pred_sub_ov=bbox_ov[pred_hoi_i['subject_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    max_ov=0
                    max_gt_id=0
                    for gt_id in range(len(gt_hoi)):
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt=min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt>max_ov:
                                max_ov=min_ov_gt
                                max_gt_id=gt_id
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] =1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>= 0.5] = 1
        iou_mat[iou_mat< 0.5] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov={}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict,match_pairs_ov

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def add_One(self,prediction):  #Add 1 to all coordinates
        for i, pred_bbox in enumerate(prediction):
            rec = pred_bbox['bbox']
            rec[0]+=1
            rec[1]+=1
            rec[2]+=1
            rec[3]+=1
        return prediction