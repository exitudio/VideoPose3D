import numpy as np

skeletons = np.load('output.npy')
print(skeletons.shape)

Data:
1. zip: /home/epinyoan/dataset/casia-b/dataset_b/zip
2. unzip (raw): /home/epinyoan/dataset/casia-b/dataset_b/dataset_b
3. frame interpolate: /home/epinyoan/dataset/casia-b/dataset_b/dataset_b_interpolate
    - 2d, 3d, 3d_rev, video
4. Not used
    - Ayman npy: /home/epinyoan/dataset/casia-b/dataset_b/npy
    - No interpolation (not finish rendering): /home/epinyoan/dataset/casia-b/dataset_b/raw_npy



# 0.
conda activate pose3Denv
# 1. unzip & remove background videos
rm *bkgrd*
# 1.1 interpolate frames for both 1 & 2
bash interpolate_frame.sh
# 1.2 separate folder
mv 1/*cl* 1_cl/

# 2. infer 2D
screen
cd git/VideoPose3D/inference/
conda activate pose3Denv
CUDA_VISIBLE_DEVICES=0 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_nm --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/1_nm
CUDA_VISIBLE_DEVICES=1 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_bg --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/1_bg
CUDA_VISIBLE_DEVICES=2 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_cl --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/1_cl

CUDA_VISIBLE_DEVICES=3 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_nm --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/2_nm
CUDA_VISIBLE_DEVICES=0 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_bg --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/2_bg
CUDA_VISIBLE_DEVICES=1 python3 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_cl --image-ext avi /home/epinyoan/dataset/casia-b/dataset_b_interpolate/video/2_cl

# 3. create custom dataset
$ cd data
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_nm -o 1_nm
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_bg -o 1_bg
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/1_cl -o 1_cl
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_nm -o 2_nm
 - 109-nm-01-144.avi.npz is broken [no joints]

python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_bg -o 2_bg
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/2_cl -o 2_cl


# 4. export 3D 
# [hack] --viz-subject need to be there with existing one of the video name
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 1_nm -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 1_bg -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 1_cl -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 2_nm -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 2_bg -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6
CUDA_VISIBLE_DEVICES=2 python3 run.py -d custom -k 2_cl -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d/ --viz-size 6

##################################################################################################################################

# With mask & reverse frame
# 1. prepare data with mask & reverse frames
python3 prepare_data_2d_custom.py -i /home/epinyoan/dataset/casia-b/dataset_b_interpolate/2d/all -o mask_rev
# 2. remove reverse frame
python3 run.py -d custom -k mask_rev -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-action custom --viz-camera 0 --viz-export /home/epinyoan/dataset/casia-b/dataset_b_interpolate/3d_mask_rev/ --viz-size 6
