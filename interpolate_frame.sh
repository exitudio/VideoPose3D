#!/bin/bash

input_dir=/home/epinyoan/dataset/casia-b/dataset_b/Datset-B-2/video/
output_dir=/home/epinyoan/dataset/casia-b/dataset_b_interpolate/2/
for entry in `ls $input_dir`
do
  echo "$input_dir$entry"
#   ffmpeg -i "$input_dir$entry" -filter "minterpolate='fps=50'" -crf 0 "$output_dir$entry"
done