# CFL
Constrained focal loss for the segmentation of anthrax spore
=======
# DeepLab-v3+ Semantic Segmentation with constrained focal loss in TensorFlow

This repo implement DeepLabv3+ with constrained focal loss in 
TensorFlow for semantic image segmentation on the anthrax spore dataset.
 The implementation is largely based on
 [rishizek's DeepLab v3+ implemantation](https://github.com/rishizek/tensorflow-deeplab-v3) 
 

## Setup
Please install latest version of TensorFlow and use Python 3.  
- For training, you need to download and extract 
[pre-trained Resnet v2 101 model](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)
from [slim](https://github.com/tensorflow/models/tree/master/research/slim)
specifying the location with `--pre_trained_model`.

## dataset
we use our anthrax spore dataset(https://drive.google.com/open?id=1-Cjy4tkhgBxTip2B_3esqw8xWkzofcZX) to train deeplab v3+ model using constrained focal loss.
in anthrax spore dataset:
    -training set:  400 images
    -testing set:  200 images
There are two classes in each image——background and anthrax spore.

## loss
We implement the proposed constrained focal loss in constrained_focal_loss_impl.py.

## Training
For training model, you first need to convert original data to
the TensorFlow TFRecord format. This enables to accelerate training seep. 
```bash
python create_pascal_tf_record.py --data_dir DATA_DIR \
                                  --image_data_dir IMAGE_DATA_DIR \
                                  --label_data_dir LABEL_DATA_DIR 
```
Once you created TFrecord for PASCAL VOC training and validation deta, 
you can start training model as follow:
```bash
python train.py --model_dir MODEL_DIR --pre_trained_model PRE_TRAINED_MODEL
```
Here, `--pre_trained_model` contains the pre-trained Resnet model, whereas 
`--model_dir` contains the trained DeepLabv3 checkpoints. 
If `--model_dir` contains the valid checkpoints, the model is trained from the 
specified checkpoint in `--model_dir`.

You can see other options with the following command:
```bash
python train.py --help
```

## Evaluation
To evaluate how model perform, one can use the following command:
```bash
python evaluate.py --help
```


## Inference
To apply semantic segmentation to your images, one can use the following commands:
```bash
python inference.py --data_dir DATA_DIR --infer_data_list INFER_DATA_LIST --model_dir MODEL_DIR 
```


## TODO:
Pull requests are welcome.
- [x] Freeze batch normalization during training
- [ ] Multi-GPU support
- [ ] Channels first support (Apparently large performance boost on GPU)
- [ ] Model pretrained on MS-COCO
- [ ] Unit test

## Acknowledgment
This repo borrows code heavily from 
- [rishizek's DeepLab v3+ implemantation](https://github.com/rishizek/tensorflow-deeplab-v3)

>>>>>>> first commit
