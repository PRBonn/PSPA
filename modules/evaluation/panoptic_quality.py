import pdb

import torch

# Props go to: https://github.com/PRBonn/HAPT/blob/master/utils/panoptic_quality.py

class PanopticQuality:
    def __init__(self):
        # initialization of a list of dictionaries with len = num classes
        self.panoptic_qualities = {}
        self.segmentation_qualities = {}
        self.recognition_qualilites = {}

    def reset(self):
        self.panoptic_qualities.clear()
        self.segmentation_qualities.clear()
        self.recognition_qualilites.clear()

    def compute_pq_single_class(self, prediction, groundtruth):
        numerator_iou = torch.tensor(0.).cuda() # IoU sum of all true positives

        class_matches = []
        unmatched_class_predictions = []

        # take the non-zero INSTANCE labels for both prediction and groundtruth
        labels = torch.unique(prediction) 
        labels = labels[labels > 0] # => instances id's

        gt_labels = torch.unique(groundtruth)
        gt_labels = gt_labels[gt_labels > 0] # => gt instances id's

        for label in labels:  # for each predicted label
            iou = torch.tensor(0.)
            best_gt_label = torch.tensor(-1) # => instance id

            mask_prediction = prediction == label
            for gt_label in gt_labels:
                mask_gt = groundtruth == gt_label

                # compute iou with all instance gt labels and store the best
                intersection = ((mask_prediction & mask_gt).sum()).float()
                union = ((mask_prediction.sum() + mask_gt.sum() )- intersection).float()
                iou_tmp = intersection / union
                if iou_tmp > iou:
                    iou = iou_tmp
                    best_gt_label = gt_label
            # if the best iou is above 0.5, store the match pred_label-gt_label-iou
            if iou > 0.5:
                result = {'pred_label': label.item(), 'gt_label': best_gt_label.item(), 'iou': iou.item()}
                class_matches.append(result)
                numerator_iou += iou
            # else, the predicted label remains unmatched (i.e., false positive)
            else:
                unmatched_class_predictions.append(label.item())

        n_true_positives = len(class_matches) # TP = number of matches
        n_false_positives = len(unmatched_class_predictions) # FP = number of unmatched predictions
        n_false_negatives = len(gt_labels) - len(class_matches) # FN = number of unmatched gt labels

        sq_one_class = numerator_iou / n_true_positives # is 'nan' if n_true_positives = 0
        rq_one_class = n_true_positives / (n_true_positives + 0.5 * (n_false_positives + n_false_negatives)) 
        pq_one_class = numerator_iou / (n_true_positives + 0.5 * n_false_positives + 0.5 * n_false_negatives)

        return pq_one_class, class_matches, sq_one_class, rq_one_class

    def compute_pq(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
        # take the semantic labels (except 0 = void label)
        semantic_gt_labels = torch.unique(semantic_gt).int()
        semantic_gt_labels = semantic_gt_labels[semantic_gt_labels > 0].int()

        if instance_pred.sum() == 0: # i.e., there aren't any predicted instances
            if instance_gt.sum() == 0: # i.e., there aren't any ground truth instances
                return 1.0, {}
            else:
                return 0.0 , {}

        for gt_label in semantic_gt_labels:
            # take pred and gt where they have sem_class = label
            semantic_tmp = (semantic_pred == gt_label).int()
            semantic_gt_tmp = (semantic_gt == gt_label).int()

            # take pred and gt instances corresponding to the considered semantic class
            instance_tmp = (instance_pred * semantic_tmp).int()
            instance_gt_tmp = (instance_gt * semantic_gt_tmp).int()

            pq, _, sq, rq = self.compute_pq_single_class(instance_tmp, instance_gt_tmp)

            # update: 
            # - count = number of images that contains the class
            # - pq is the overall panoptic quality of the images
            if gt_label.item() not in self.panoptic_qualities.keys():
                self.panoptic_qualities[gt_label.item()] = {}
                # self.segmentation_qualities[gt_label.item()] = {}
                self.recognition_qualilites[gt_label.item()] = {}

                self.panoptic_qualities[gt_label.item()]['count'] = 1
                self.panoptic_qualities[gt_label.item()]['pq'] = pq.item()
                self.recognition_qualilites[gt_label.item()]['rq'] = rq
            if gt_label.item() not in self.segmentation_qualities.keys():
                if not sq.isnan():
                    self.segmentation_qualities[gt_label.item()] = {}
                    self.segmentation_qualities[gt_label.item()]['count'] = 1
                    self.segmentation_qualities[gt_label.item()]['sq'] = sq.item()
            else:
                self.panoptic_qualities[gt_label.item()]['count'] += 1
                self.panoptic_qualities[gt_label.item()]['pq'] = ((self.panoptic_qualities[gt_label.item()]['pq'] * (self.panoptic_qualities[gt_label.item()]['count'] - 1) + pq) / self.panoptic_qualities[gt_label.item()]['count']).item()
                self.recognition_qualilites[gt_label.item()]['rq'] = ((self.recognition_qualilites[gt_label.item()]['rq'] * (self.panoptic_qualities[gt_label.item()]['count'] - 1) + rq) / self.panoptic_qualities[gt_label.item()]['count'])
                if not sq.isnan():
                    self.segmentation_qualities[gt_label.item()]['count'] += 1
                    self.segmentation_qualities[gt_label.item()]['sq'] = ((self.segmentation_qualities[gt_label.item()]['sq'] * (self.segmentation_qualities[gt_label.item()]['count'] - 1) + sq) / self.segmentation_qualities[gt_label.item()]['count']).item()
              
        panoptic_quality = self.average_pq(self.panoptic_qualities)
        #segmentation_quality = self.average_sq(self.segmentation_qualities)
        #recognition_quality = self.average_rq(self.recognition_qualilites)

        return panoptic_quality, self.panoptic_qualities

    def average_pq(self, panoptic_qualities):
        # overall panoptic quality averaged from all classes
        pq = 0.
        n = 0
        for item in panoptic_qualities.keys():
            pq += panoptic_qualities[item]['pq']
            n += 1
        if n == 0:
            return -1
        pq /= n
        return pq
    
    def average_sq(self, segmentation_qualities):
        # overall segmentation quality averaged from all classes
        sq = 0.
        n = 0
        for item in segmentation_qualities.keys():
            sq += segmentation_qualities[item]['sq']
            n += 1
        if n == 0:
            return -1
        sq /= n
        return sq
    
    def average_rq(self, recognition_qualilites):
        # overall recognition quality averaged from all classes
        rq = 0.
        n = 0
        for item in recognition_qualilites.keys():
            rq += recognition_qualilites[item]['rq']
            n += 1
        if n == 0:
            return -1
        rq /= n
        return rq

    def panoptic_quality_forward(self, batch_semantic_pred, batch_semantic_gt, batch_instance_pred, batch_instance_gt):
        batch_size = batch_semantic_pred.shape[0]
        for item in range(batch_size):
            semantic_gt = batch_semantic_gt[item]
            semantic_pred = batch_semantic_pred[item]
            instance_gt = batch_instance_gt[item]
            instance_pred = batch_instance_pred[item]
            # for each batch item
            pq, pqs = self.compute_pq(
                semantic_pred, semantic_gt, instance_pred, instance_gt)
        # return pq, pqs

class PanopticEvaluator():
  def __init__(self):
    self.evaluator = PanopticQuality()

  def update(self, batch_semantic_pred, batch_semantic_gt, batch_instance_pred, batch_instance_gt):
    self.evaluator.panoptic_quality_forward(batch_semantic_pred, batch_semantic_gt, batch_instance_pred, batch_instance_gt)

  def compute(self):
    pq = self.evaluator.average_pq(self.evaluator.panoptic_qualities)
    sq = self.evaluator.average_sq(self.evaluator.segmentation_qualities)
    rq = self.evaluator.average_rq(self.evaluator.recognition_qualilites)
    return pq, sq, rq

  def reset(self):
    self.evaluator.reset()

  @property
  def results_pq_classwise(self):
    return self.evaluator.panoptic_qualities
  
  @property
  def results_sq_classwise(self):
    return self.evaluator.segmentation_qualities
  
  @property
  def results_rq_classwise(self):
    return self.evaluator.recognition_qualilites