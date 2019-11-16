#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

# caffe/build/tools的位置
TOOLS=/home/wurui/idcard/caffe_ocr_for_linux/build/tools

# 生成的LMDB文件位置
EXAMPLE=/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0
# EXAMPLE=/home/wurui/idcard/Synthetic_Chinese_String_Dataset

# train.txt 和 test.txt 的位置
DATA=/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0
# DATA=/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset

# 训练集,测试集 图片的位置, 最后的 '/' 别漏了
# TRAIN_DATA_ROOT=/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset/images/
# VAL_DATA_ROOT=/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset/images/
TRAIN_DATA_ROOT=/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/images/
VAL_DATA_ROOT=/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/images/

# 删除已经存在的
rm -rf $EXAMPLE/idcard_train_lmdb
rm -rf $EXAMPLE/idcard_test_lmdb

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=32
  RESIZE_WIDTH=180
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=1 \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/idcard_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=1 \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/idcard_test_lmdb

echo "Done."
