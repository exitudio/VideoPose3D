# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import sys

import argparse
from data_utils import suggest_metadata

output_prefix_2d = 'data_2d_custom_'

def add_reverse(keypoints):
    flipped = np.flip(keypoints, axis=0)
    half_idx = int(flipped.shape[0]/2)
    prepend = flipped[half_idx:]
    append = flipped[:half_idx]
    return np.concatenate((prepend, keypoints, append))

def get_first_repeat_true(mask):
    for i in range(mask.shape[0]-1):
        if mask[i] == mask[i+1] and mask[i]:
            return i
    return 100000
    
def get_mask_by_prob(skeletons):
    kps_prob = skeletons[:,:,3].mean(axis=1)
    mask = kps_prob>.12
    first_id = get_first_repeat_true(mask)
    last_id = mask.shape[0] - get_first_repeat_true(np.flip(mask))
    mask = np.zeros_like(mask)
    mask[first_id:last_id] = True
    return mask

def mask_n_reverse(data, video_metadata):
    keypoints = data[0]['keypoints']
    mask = get_mask_by_prob(keypoints)
    keypoints = add_reverse(keypoints[mask])
    keypoints = keypoints[:,:,:2]
    return [{'keypoints': keypoints}], video_metadata


def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print('Processing {}'.format(filename))
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)
        
    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :4] # Extract (x, y)
    
    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
    
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')
    
    return [{
        'start_frame': 0, # Inclusive
        'end_frame': len(kp), # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Custom dataset creator')
    parser.add_argument('-i', '--input', type=str, default='', metavar='PATH', help='detections directory')
    parser.add_argument('-o', '--output', type=str, default='', metavar='PATH', help='output suffix for 2D detections')
    args = parser.parse_args()
    
    if not args.input:
        print('Please specify the input directory')
        exit(0)
        
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)
    
    print('Parsing 2D detections from', args.input)
    
    metadata = suggest_metadata('coco')
    metadata['video_metadata'] = {}
    
    output = {}
    file_list = glob(args.input + '/*.npz')
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data, video_metadata = mask_n_reverse(*decode(f)) # need to reverse line71 to "kp[:, :, :2]"
        if data[0]['keypoints'].shape[0] == 0:
            continue
        output[canonical_name] = {}
        output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]
        metadata['video_metadata'][canonical_name] = video_metadata

    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')