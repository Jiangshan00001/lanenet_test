# lanenet_test
torch implementation for paper: Towards End-to-End Lane Detection an Instance Segmentation Approach
this is a little program for lane detect.

## get dataset: TUSIMPLE

!mkdir -p /data/tusimple
!wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
!unzip train_set.zip -d /data/tusimple  

## train:
python3 train.py
the gpu is needed for train.

## run:

run.py


## pics:


![inst distance](pics/instance_logit_distnace.png)
![lanenet loss](pics/lanenet_loss.png)
![run image result](pics/run_resultimage.png)
![train draw test](pics/draw_test.png)
