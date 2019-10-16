#!/usr/bin/env sh
# convert images to lmdb

DATA=/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset
IMGDIRNAME=images
IMGLIST=train.txt
LMDBNAME=train_lmdb

rm -rf $DATA/$LMDBNAME
echo 'converting images...'
/home/wurui/project/easy-pva/caffe-fast-rcnn/build/tools/convert_imageset --shuffle=true \
$DATA/$IMGDIRNAME/ $DATA/$IMGLIST $DATA/$LMDBNAME