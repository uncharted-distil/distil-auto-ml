# -*- coding: utf-8 -*-
# file: d3mds.py
# lab: MIT Lincoln Lab
# author(s): sw26425
# description: a rudimentary API for interacting with D3MDataSupply, which mainly consists of a Dataset and a Problem

import os, json, sys
import pandas as pd
import numpy as np
import warnings

DATASET_SCHEMA_VERSION = '3.0'
PROBLEM_SCHEMA_VERSION = '3.0'

class D3MDataset:
    dsHome = None
    dsDoc = None
    learningDataFile = None

    def __init__(self, datasetPath):
        self.dsHome = datasetPath

        # read the schema in dsHome
        _dsDoc = os.path.join(self.dsHome, 'datasetDoc.json')
        assert os.path.exists(_dsDoc)
        with open(_dsDoc, 'r') as f:
            self.dsDoc = json.load(f)

        # # make sure the versions line up
        # if self.get_datasetSchemaVersion() != DATASET_SCHEMA_VERSION:
        #   warnings.warn("the datasetSchemaVersions in the API and datasetDoc do not match !!!!!!!")

        # locate the special learningData file
        self.learningDataFile = self._get_learning_data_path()

    def get_datasetID(self):
        """
        Returns the datasetID from datasetDoc
        """
        return self.dsDoc['about']['datasetID']

    def get_datasetSchemaVersion(self):
        """
        Returns the dataset schema version that was used to create this dataset
        """
        return self.dsDoc['about']['datasetSchemaVersion']

    def get_learning_data(self, view=None, problem=None):
        """
        Returns the contents of learningData.doc as a DataFrame.
        If view is 'TRAIN' or 'TEST', then the full learningData is filtered to return learningData only for that view.
        For view-based filtering, the problem object has to be passed because this method used the splitsData from the problem.
        """
        df = pd.read_csv(self.learningDataFile, index_col='d3mIndex')

        if view is None:
            return df

        if view.upper() == 'TRAIN' or view.upper() == 'TEST':
            if problem is None:
                raise RuntimeError('asking for learningData for a split, but the problem is not given') 
            splitsdf = problem.get_datasplits(view)
            df = df.loc[splitsdf.index]
            return df

    def get_learning_data_columns(self):
        res = self._get_learning_data_resource()
        return res['columns']


    def set_learning_data(self, df):
        """
        Sets the contents of the learningData file to df
        """
        df.to_csv(self.learningDataFile)


    def delete_column_entries(self, target):
        """
        Deletes all the entries of a particular column of a particular tabular data resource.
        The deleted entries are set to numpy.NaN
        """
        resID = target['resID']
        colIndex = target['colIndex']
        colName = target['colName']

        for res in self.dsDoc['dataResources']:
            _resID = res['resID']
            if _resID != resID:
                continue
            _resPath = res['resPath']
            _resPath = os.path.join(self.dsHome, _resPath)
            _resType = res['resType']
            assert _resType == 'table'
            for col in res['columns']:
                _colIndex = col['colIndex']
                if _colIndex != colIndex:
                    continue
                _colName = col['colName']
                assert _colName == colName
                df = pd.read_csv(_resPath)
                df[_colName] = [np.NaN]*len(df[_colName])
                df.to_csv(_resPath, index=None)
                return True
            raise RuntimeError('could not find the column') 
        raise RuntimeError('could not find the resource')

    def delete_identifying_fields(self, view):
        """
        Deletes some fields that might contain identifying information. 
        These fields should not be in the train or test view during the blinds evaluation.
        """
        assert view.upper()=='TRAIN' or view.upper()=='TEST' # ensures we perform this only if view is train or test
        
        self.dsDoc['about']['datasetName']='NULL'
        self.dsDoc['about']['redacted'] = True
        
        try:
            del self.dsDoc['about']['description']
        except KeyError:
            pass
        try:
            del self.dsDoc['about']['citation']
        except KeyError:
            pass
        try:
            del self.dsDoc['about']['source']
        except KeyError:
            pass
        try:
            del self.dsDoc['about']['sourceURI']
        except KeyError:
            pass
        
        # save datasetDoc.json file
        with open(os.path.join(self.dsHome, 'datasetDoc.json'), 'w') as fp:
            json.dump(self.dsDoc, fp, indent=2, sort_keys=False)

    def open_raw_timeseries_file(self, filename):
        filepath = os.path.join(self.dsHome, 'timeseries', filename)
        assert os.path.exists(filepath)
        return open(filepath)

    ############# private methods 
    def _get_learning_data_path(self):
        """
        Returns the path of learningData.csv in a dataset
        """
        for res in self.dsDoc['dataResources']:
            resID = res['resID']
            resPath = res['resPath']
            resType = res['resType']
            resFormat = res['resFormat']
            
            dirname = os.path.basename(os.path.normpath(os.path.dirname(resPath)))

            if resType =='table' and dirname=='tables':
                if 'learningData.csv' in res['resPath'] :
                    return os.path.join(self.dsHome, resPath)
                else:
                    # raise RuntimeError('non-CSV learningData (not implemented yet ...)')      
                    continue
        # if the for loop is over and learningDoc is not found, then return None
        raise RuntimeError('could not find learningData file the dataset')

    def _get_learning_data_resource(self):
        """
        Returns the path of learningData.csv in a dataset
        """
        for res in self.dsDoc['dataResources']:
            resID = res['resID']
            resPath = res['resPath']
            resType = res['resType']
            resFormat = res['resFormat']
            if resType =='table':
                if 'learningData.csv' in res['resPath'] :
                    return res
                else:
                    raise RuntimeError('could not find learningData.csv')       
        # if the for loop is over and learningDoc is not found, then return None
        raise RuntimeError('could not find learningData resource')


class D3MProblem:
    prHome = None
    prDoc = None
    splitsFile = None

    def __init__(self, problemPath):
        self.prHome = problemPath

        # read the schema in prHome
        _prDoc = os.path.join(self.prHome, 'problemDoc.json')
        assert os.path.exists(_prDoc)
        with open(_prDoc, 'r') as f:
            self.prDoc = json.load(f)

        # make sure the versions line up
        if self.get_problemSchemaVersion() != PROBLEM_SCHEMA_VERSION:
            warnings.warn("the problemSchemaVersions in the API and datasetDoc do not match !!!!!!!")

        # locate the splitsFile
        self.splitsFile = self._get_datasplits_file()

    def get_problemID(self):
        """
        Returns the problemID from problemDoc
        """
        return self.prDoc['about']['problemID']

    def get_problemSchemaVersion(self):
        """
        Returns the problem schema version that was used to create this dataset
        """
        return self.prDoc['about']['problemSchemaVersion']

    def get_datasetID(self):
        """
        Returns the ID of the dataset referenced in the problem 
        """
        return self.prDoc['inputs']['data'][0]['datasetID']

    def get_targets(self):
        """
        Looks at the problemDoc and returns the colIndex and colName of the target variable
        """
        return self.prDoc['inputs']['data'][0]['targets']

    def get_datasplits(self, view=None):
        """
        Returns the data splits splits in a dataframe
        """
        df = pd.read_csv(self.splitsFile, index_col='d3mIndex')
        
        if view is None:
            return df
        elif view.upper() == 'TRAIN':
            df = df[df['type']=='TRAIN']
            return df
        elif view.upper() == 'TEST':
            df = df[df['type']=='TEST']
            return df

    def set_datasplits(self, df):
        """
        Sets the contents of the dataSplits file to df
        """
        df.to_csv(self.splitsFile)

    def delete_identifying_fields(self, view):
        """
        Deletes some fields that might contain identifying information. 
        These fields should not be in the train or test view during the blinds evaluation.
        """
        assert view.upper()=='TRAIN' or view.upper()=='TEST' # ensures we perform this only if view is train or test
        
        self.prDoc['about']['problemName']='NULL'
        try:
            del self.prDoc['about']['problemDescription']
        except KeyError:
            pass
        
        # save datasetDoc.json file
        with open(os.path.join(self.prHome, 'problemDoc.json'), 'w') as fp:
            json.dump(self.prDoc, fp, indent=2, sort_keys=False)

    def get_performance_metrics(self):
        return self.prDoc['inputs']['performanceMetrics']

    ############# private methods 
    def _get_datasplits_file(self):
        splitsFile = self.prDoc['inputs']['dataSplits']['splitsFile']
        splitsFile = os.path.join(self.prHome, splitsFile)
        assert os.path.exists(splitsFile)
        return splitsFile


class D3MDS:
    dataset = None
    problem = None
    
    def __init__(self, datasetPath, problemPath):
        self.dataset = D3MDataset(datasetPath) 
        self.problem = D3MProblem(problemPath)
        # sanity check
        assert self.dataset.get_datasetID() == self.problem.get_datasetID()

    def _get_target_columns(self, df):
        target_cols = []
        targets = self.problem.get_targets()
        for target in targets:
            colIndex = target['colIndex']-1 # 0th column is d3mIndex
            colName = df.columns[colIndex]
            assert colName == target['colName']
            target_cols.append(colIndex)
        return target_cols

    def get_data_all(self):
        df = self.dataset.get_learning_data(view=None, problem=None)
        return df

    def get_train_data(self):
        df = self.dataset.get_learning_data(view='train', problem=self.problem)
        target_cols = self._get_target_columns(df)
        df.drop(df.columns[target_cols],axis=1,inplace=True)
        return df

    def get_train_targets(self):
        df = self.dataset.get_learning_data(view='train', problem=self.problem)
        target_cols = self._get_target_columns(df)
        X = df.shape[0]
        Y = len(target_cols)
        return (df[df.columns[target_cols]]).values.reshape(X,Y)
        # return np.ravel(df[df.columns[target_cols]])
        
    def get_test_data(self):
        df = self.dataset.get_learning_data(view='test', problem=self.problem)
        target_cols = self._get_target_columns(df)
        df.drop(df.columns[target_cols],axis=1,inplace=True)
        return df

    def get_test_targets(self):
        df = self.dataset.get_learning_data(view='test', problem=self.problem)
        target_cols = self._get_target_columns(df)
        X = df.shape[0]
        Y = len(target_cols)
        return (df[df.columns[target_cols]]).values.reshape(X,Y)
        # return np.ravel(df[df.columns[target_cols]])


# --

import numpy as np
import logging

def group_gt_boxes_by_image_name(gt_boxes):
    '''
    This function takes a list of ground truth boxes and turn them into a
    dict mapping an image name to an array containing the 4 coordinates of 
    the edges delimiting a bounding box 

    Parameters:
    -----------
    gt_boxes: list
     List of ground truth boxes. Each box is represented as a list with the
     following format: [image_name, x_min, y_min, x_max, y_max].

    Returns:
    --------
    gt_dict: dict
     Dictionary mapping every image name to an array of the bounding boxes:
        {'image_name' : [
                {'bbox': [x_min, y_min, x_max, y_max]},
                {'bbox': [x_min, y_min, x_max, y_max]},
                ...
            ]
        }

    '''
    gt_dict = {}

    for box in gt_boxes:

        image_name = box[0]
        bbox = box[1:]

        if image_name not in gt_dict.keys():
            gt_dict[image_name] = []

        gt_dict[image_name].append({'bbox': bbox})

    return gt_dict


def unvectorize(targets):
    """
    If ``targets`` have two columns (index, object detection target) or three (index, object detection
    target, confidence), we make it into 5 or 6, respectively, by splitting the second column into
    4 columns for each bounding box edge.
    
    Parameters:
    -----------
    targets: list
     List of bounding boxes. Each box is represented as a list with the
     following format: 
        
        Case 1 (confidence provided):
             ['image_name', 'x_min, 'y_min, x_max, y_max', 'confidence']
        Case 2 (confidence not provided):
             ['image_name', 'x_min, 'y_min, x_max, y_max']
        Case 3: (List with more than three elements) 
            ['image_name', ... ]

    Returns:
    --------
    new_targets: list
        List following the following format:
    
         Case 1 (confidence provided): 
             ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence']
         Case 2 (confidence not provided):
             ['image_name', 'x_min', 'y_min', 'x_max', 'y_max']
         Case 3: (List with more than three elements) 
            ['image_name', ... ]
        


    """

    new_targets = []

    for target in targets:
        if len(target) == 2:
            new_targets.append([target[0]] + target[1].split(','))
        elif len(target) == 3:
            new_targets.append([target[0]] + target[1].split(',') + list(target[2:]))
        else:
            new_targets.append(target)

    return new_targets



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def objectDetectionAP(dets,
                      gts,
                      ovthresh=0.5,
                      use_07_metric=False):
    """
    This function takes a list of ground truth boxes and a list of detected bounding boxes
    for a given class and computes the average precision of the detections with respect to
    the ground truth boxes.

    Parameters:
    -----------
    dets: list
     List of bounding box detections. Each box is represented as a list
     with format:
         Case 1 (confidence provided):
             ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence']
         Case 2 (confidence not provided):
             ['image_name', 'x_min', 'y_min', 'x_max', 'y_max']
         Case 3 (confidence provided, coordinates as string):
             ['image_name', 'x_min, 'y_min, x_max, y_max', 'confidence']
         Case 4 (confidence not provided, coordinates as string):
             ['image_name', 'x_min, 'y_min, x_max, y_max']

    gts: list
     List of ground truth boxes. Each box is represented as a list with the
     following format: [image_name, x_min, y_min, x_max, y_max].

    [ovthresh]: float
     Overlap threshold (default = 0.5)

    [use_07_metric]: boolean
     Whether to use VOC07's 11 point AP computation (default False)

    Returns:
    --------
    rec: 1d array-like
     Array where each element (rec[i]) is the recall when considering i+1 detections

    prec: 1d array-like
     Array where each element (rec[i]) is the precision when considering i+1 detections

    ap: float
     Average precision between detected boxes and the ground truth boxes.
     (it is also the area under the precision-recall curve).

    Example:

    With confidence scores:
    >> predictions_list = [['img_00285.png',330,463,387,505,0.0739],
                           ['img_00285.png',420,433,451,498,0.0910],
                           ['img_00285.png',328,465,403,540,0.1008],
                           ['img_00285.png',480,477,508,522,0.1012],
                           ['img_00285.png',357,460,417,537,0.1058],
                           ['img_00285.png',356,456,391,521,0.0843],
                           ['img_00225.png',345,460,415,547,0.0539],
                           ['img_00225.png',381,362,455,513,0.0542],
                           ['img_00225.png',382,366,416,422,0.0559],
                           ['img_00225.png',730,463,763,583,0.0588]]
    >> ground_truth_list = [['img_00285.png',480,457,515,529],
                            ['img_00285.png',480,457,515,529],
                            ['img_00225.png',522,540,576,660],
                            ['img_00225.png',739,460,768,545]]

    >> rec, prec, ap = objectDetectionAP(predictions_list, ground_truth_list)
    >> print(ap)
    0.125

    Without confidence scores:
    >> predictions_list = [['img_00285.png',330,463,387,505],
                           ['img_00285.png',420,433,451,498],
                           ['img_00285.png',328,465,403,540],
                           ['img_00285.png',480,477,508,522],
                           ['img_00285.png',357,460,417,537],
                           ['img_00285.png',356,456,391,521],
                           ['img_00225.png',345,460,415,547],
                           ['img_00225.png',381,362,455,513],
                           ['img_00225.png',382,366,416,422],
                           ['img_00225.png',730,463,763,583]]
    >> ground_truth_list = [['img_00285.png',480,457,515,529],
                            ['img_00285.png',480,457,515,529],
                            ['img_00225.png',522,540,576,660],
                            ['img_00225.png',739,460,768,545]]

    >> rec, prec, ap = objectDetectionAP(predictions_list, ground_truth_list)
    >> print(ap)
    0.0625

    """

    # Unvectorize the detected bounding boxes
    dets = unvectorize(dets)
    gts = unvectorize(gts)

    # Load ground truth
    gt_dict = group_gt_boxes_by_image_name(gts)

    # extract gt objects for this class
    recs = {}
    npos = 0

    imagenames = sorted(gt_dict.keys())
    for imagename in imagenames:
        R = [obj for obj in gt_dict[imagename]]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        recs[imagename] = {'bbox': bbox,
                           'det': det}

    # Load detections
    det_length = len(dets[0])

    # Check that all boxes are the same size
    for det in dets:
        assert len(det) == det_length, 'Not all boxes have the same dimensions.'



    image_ids = [x[0] for x in dets]
    BB = np.array([[float(z) for z in x[1:5]] for x in dets])

    if det_length == 6:
        logging.info('confidence scores are present')
        confidence = np.array([float(x[-1]) for x in dets])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

    else:
        logging.info('confidence scores are not present')
        num_dets = len(dets)
        sorted_ind = np.arange(num_dets)
        sorted_scores = np.ones(num_dets)

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # print('sorted_ind: ', sorted_ind)
    # print('sorted_scores: ', sorted_scores)
    # print('BB: ', BB)
    # print('image_ids: ', image_ids)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        # print('det %d: ' % d)
        # print('bb: ', bb)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            # print('overlaps: ', overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                # print('Box matched!')
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                # print('Box was already taken!')
                fp[d] = 1.
        else:
            # print('No match with sufficient overlap!')
            fp[d] = 1.

    # print('tp: ', tp)
    # print('fp: ', fp)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


