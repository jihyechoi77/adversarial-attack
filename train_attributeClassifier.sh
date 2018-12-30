#!/bin/bash


# VGG16
# python model_train.py --epoch 9
python model_evaluate.py

python model_train.py --loss 'binary_crossentropy_custom' \
--epoch 9 \ 
--save_name 'VGG16-dim160-attrClassifier-customloss'
python model_evaluate.py --model_name 'VGG16-dim160-attrClassifier-customloss'


# InceptionV3
python model_train.py --model_type InceptionV3 --save_name 'InceptionV3-dim160-attrClassifier' --epoch 20
python model_evaluate.py --model_name 'InceptionV3-dim160-attrClassifier'

python model_train.py --model_type InceptionV3 --save_name 'InceptionV3-dim160-attrClassifier-customloss' --epoch 20
python model_evaluate.py --model_name 'InceptionV3-dim160-attrClassifier-customloss'

: '
# InceptionResNetV2
# python model_train.py --model_type InceptionResNetV2 --save_name 'InceptionResNetV2-dim160-attrClassifier' --epoch 50
# python model_evaluate.py --model_name 'InceptionResNetV2-dim160-attrClassifier'

# python model_train.py --model_type InceptionResNetV2 --save_name 'InceptionResNetV2-dim160-attrClassifier-customloss' --epoch 30
# python model_evaluate.py --model_name 'InceptionResNetV2-dim160-attrClassifier-customloss'


python model_train.py --data_dir '/home/jihyec/data/vgg/vgg-aligned-dim160-tightCrop-ready' \
--model_type 'VGG16' \
--loss 'categorical_crossentropy' \
--save_name 'VGG16-dim160-faceClassifier' \
--epoch 10 \
--steps_per_epoch 5000 \
--train_multilab False \
--batch_size 256


python model_train.py --data_dir '/home/jihyec/data/vgg/vgg-aligned-dim160-tightCrop-ready' \
--model_type 'inceptionV3' \
--loss 'categorical_crossentropy' \
--save_name 'inceptionV3-dim160-faceClassifier' \
--epoch 10 \
--steps_per_epoch 300 \
--train_multilab False \
--batch_size 512
'
