#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

# 生成的LMDB文件位置
EXAMPLE=/home/wurui/idcard/data/TrainCRNN-prob0.9-ratio3.0
# train.txt 和 test.txt 的位置
DATA=/home/wurui/idcard/data/TrainCRNN-prob0.9-ratio3.0
# caffe/build/tools的位置
TOOLS=/home/wurui/idcard/caffe_ocr_for_linux/build/tools

# 训练集  测试集 位置, 最后的 '/' 别漏了
TRAIN_DATA_ROOT=/home/wurui/idcard/data/TrainCRNN-prob0.9-ratio3.0/img/part5/
VAL_DATA_ROOT=/home/wurui/idcard/data/TrainCRNN-prob0.9-ratio3.0/img/part5/

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
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/idcard_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/idcard_test_lmdb

echo "Done."
