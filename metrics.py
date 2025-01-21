#! /dls/ebic/data/staff-scratch/samclark/envs/picking/bin/python

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import argparse

from topaz.metrics import precision_recall_curve
from topaz.utils.conversions import boxes_to_coordinates


parser = argparse.ArgumentParser(
    prog='picking_metrics',
    description='Orientation determination model'
    )

def add_arguments(parser):
    parser.add_argument('-fp','--files-predicted', help='path to predicted box files')
    parser.add_argument('-ft','--files-truth', help='path to groundtruth box files')
    parser.add_argument('-o','--outfile', help='path to groundtruth box files',default=None)

    parser.add_argument('-r', '--assignment-radius', required=True, type=int, help='maximum distance between prediction and labeled target allowed for considering them a match')
    return parser

def IOU(dists,radius):
  
    dists = np.min(dists,axis = 0)
    dists[ np.isinf(dists) ] = radius

    intersection = 2*radius*radius*np.arccos(np.divide(dists,2*radius)) \
    - np.multiply(dists,np.sqrt(radius*radius - (dists/2)**2))

    iou_arr = intersection/(2*np.pi*radius*radius-intersection) 

    iou = np.sum(iou_arr)

    return iou/float(len(dists))

def DICE(dists,radius):

    dists = np.min(dists,axis = 0)
    dists[ np.isinf(dists) ] = radius

    intersection = 2*radius*radius*np.arccos(np.divide(dists,2*radius)) \
    - np.multiply(dists,np.sqrt(radius*radius - (dists/2)**2))

    dice_arr = intersection/(np.pi*radius*radius) 

    dice = np.sum(dice_arr)

    return dice/float(len(dists))

def match_coordinates(targets, preds, radius):
    d1 = (preds[:,np.newaxis] - targets[np.newaxis])**2


    d2 = np.sum(d1, 2)

    cost1 = d1 - radius*radius
    cost = d2 - radius*radius/4
    d2[d2 > radius*radius/4] = np.inf

    d = np.sqrt(d2)

    pred_tally = np.count_nonzero(~np.isinf(d),axis = 0)
    targ_tally = np.sum(~np.isinf(d),axis = 1)

    TP = np.count_nonzero(targ_tally)
    FN = cost.shape[0] - TP
    FP = cost.shape[1] - np.count_nonzero(pred_tally)
    
    precision = TP/float(TP+FP)
    recall = TP/float(TP+FN)

    if precision == 0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall) 

    iou = IOU(d,radius)

    dice = DICE(d, radius)

    return precision, recall, F1, iou, dice

def read_cbox(file_in):

    lines = [p[:-1] for p in open(file_in, 'r').readlines()]
    lines = [p for p in lines if '<NA>' in p]
   
    arr_in = []
    for l in lines:

        entry = np.array(l.split(' '))
        if len(arr_in) == 0:

            arr_in = entry
        else:
            arr_in = np.vstack((arr_in,entry))

    if len(arr_in) == 0:
        return arr_in
    else:
        arr_in[arr_in=='<NA>'] = '0'
        if arr_in.ndim == 1: 
            arr_in = arr_in[np.newaxis]
        return arr_in.astype(float)

def cbox_to_coord(data):

    coords = []



    for i in range(len(data)):


        coord_tmp  = np.array([data[i,0]+data[i,3]/2,
        data[i,1]+data[i,4]/2])
        #print(coord_tmp.ndim)

        if len(coords) == 0:

            coords = coord_tmp 

        else:

            coords = np.vstack((coords, coord_tmp))

    if coords.ndim == 1:
        coords = coords[np.newaxis]


    return coords
	

def main(args):

    Metrics(args.files_truth, args.files_pred, args.outfile,args.assignment_radius)

def Metrics(truth_dir, pred_dir, match_radius, outfile,average_file):

    files_truth = glob(truth_dir+'*')

    files_pred = glob(pred_dir+'*')


    if len(files_pred) == 0:
        return None 

    prec_arr = []
    rec_arr = []
    f1_arr = []
    iou_arr = []
    dice_arr = []
    diff_arr = []
    size_arr = []
    files_pred = []

    for filet in files_truth:

        file_id = filet.split('/')[-1].split('.')[0]
        filep = [i for i in glob(pred_dir+'*') if file_id in i ]

        if len(filep) == 0:
            continue

        filep = filep[0]
        files_pred.append(filep)
        try:
            target_data = pd.read_csv(filet, sep = '\t', header=None)
            predict_data = read_cbox(filep)

            if len(predict_data) == 0:

                prec_arr.append(0.)
                rec_arr.append(0.)
                f1_arr.append(0.)
                iou_arr.append(0.)
                dice_arr.append(0.)
                diff_arr.append(0.)
                continue
                
            targets = boxes_to_coordinates(target_data.values)
            predicts = cbox_to_coord(predict_data)
            
            if len(size_arr) == 0:
                size_arr = predict_data[:,6]
            else:
                size_arr = np.append(size_arr,predict_data[:,6])
            targets = targets.astype(float)
            predicts = predicts.astype(float)
            #print(predicts)
            #print(targets)
            precision, recall, F1, iou, dice = match_coordinates(targets, predicts, match_radius)
            prec_arr.append(precision)
            rec_arr.append(recall)
            f1_arr.append(F1)
            iou_arr.append(iou)
            dice_arr.append(dice)
            diff_arr.append(float(len(predict_data)-len(target_data)))

        except pd.errors.EmptyDataError:
            
            prec_arr.append(0.)
            rec_arr.append(0.)
            f1_arr.append(0.)
            iou_arr.append(0.)
            dice_arr.append(0.)
            diff_arr.append(0.)
            continue
            print('CSV file is empty')

    #print(size_arr.shape)
    print(np.mean(size_arr))
    print(len(files_pred))
    print(len(prec_arr))
    print(len(rec_arr))
    print(len(f1_arr))
    print(len(iou_arr))
    print(len(dice_arr))
    print(len(diff_arr))

    met_df = pd.DataFrame({'file':files_pred, 'precision':prec_arr, 'recall':rec_arr, 
                'F1':f1_arr,'IOU':iou_arr, 'dice':dice_arr, 'diff': diff_arr})

    if outfile:
        met_df.to_csv(str(outfile)+'_metrics.csv',index=False)
    else:
        met_df.to_csv('metrics.csv',index=False)

    met_vals = met_df.values[:,1:].astype(float)
    print(met_vals.shape)
    met_av = np.mean(met_vals, axis=0)
    met_std = np.std(met_vals, axis=0)

    with open(average_file+".csv", "a+") as av_file:

        av_file.write(str(outfile)+','+','.join(met_av.astype(str))+','+','.join(met_std.astype(str))+'\n')


if __name__ == '__main__':

    parser = add_arguments(parser)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(
            'Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(
                unparsed))
        exit(0)

    main(args)
