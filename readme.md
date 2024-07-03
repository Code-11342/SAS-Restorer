# Automatic Maxilla Completion with Stochastic Anomaly Simulation
Implementation of paper - Stochastic Anomaly Simulation for Maxilla Completion from Cone-Beam Computed Tomography

## Setup
The code has been tested under environment with Python=3.8.8 and PyTorch=1.10.0+cu113.

Install the following packages with pip:
```
pip install -r requirements.txt
```

Install the U-Net3d module code:
```
cd model/unet3D/pytorch-3dunet
python setup.py --install
```

Reference: The code for U-Net3D module at `./model/unet3D` is modified from the repository [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet.git). The code for non-rigid data augmentation at `./dataset/mask_gen/volumentations` is modified from the repository [volumentations](https://github.com/ZFTurbo/volumentations.git).

## Testing
We provide the test code and example data for automatic maxillary restoration and cleft defect mask prediction. 

The trained model weights can be downloaded from [here](https://drive.google.com/drive/folders/1J5l0tCkwVb2deuS6vBaOfItZ_nvrrGQS?usp=sharing).

The downloaded `best_inp_model.pth` file contains the weight of the CBCT restoration model, and should be placed in `../save_dir/inp_model`. 

The downloaded `best_unet_module.pytorch` file contains the weight of the cleft defect mask prediction module, and should be placed in `../save_dir/unet_module`.

Put the example test images in `../test_dir/input_data/image` directory, and put the corresponding test masks in `--input_mask_dir=../test_dir/input_data/mask`.

Run the following command:
```
CUDA_VISIBLE_DEVICES=0 python test.py --gpu_id=0 \
    --inp_model_path=../save_dir/inp_model/best_inp_model.pth \
    --unet_module_path=../save_dir/unet_module/best_unet_module.pytorch \
    --input_image_dir=../test_dir/input_data/image \
    --input_mask_dir=../test_dir/input_data/mask \
    --output_image_dir=../test_dir/output_data/output_image \
    --output_label_dir=../test_dir/output_data/output_label
```
The predicted result will appear in `../test_dir/output_data`.

## Training
We provide the training code. You can train the model with the following commands step by step.

### Train Stage 1
In the first training stage, you can train the model on the synthetic dataset to initialize the inpainting model:
```
python train_stage_1.py \
    --model_name=$MODEL_NAME$ \
    --com_dataset_dir=$COM_DATASET_DIR$ \
    --save_dir=$SAVE_DIR$ \
    --log_interval=1 --sample_interval=1 \
    --save_interval=2 --batch_size=1
```
`$COM_DATASET_DIR$` is the path to the dataset directory. The structure of the dataset directory should be as follows:

train <br />
├── image <br />
│&emsp;├── train_data_1.nii.gz <br />
│&emsp;├── train_data_2.nii.gz <br />
│&emsp;└── ... <br />
├── mask <br />
│&emsp;├── train_data_1.nii.gz <br />
│&emsp;├── train_data_2.nii.gz <br />
│&emsp;└── ... <br />
└── maxi <br />
&nbsp;&nbsp;&nbsp;&emsp;├── train_data_1.nii.gz <br />
&nbsp;&nbsp;&nbsp;&emsp;├── train_data_2.nii.gz <br />
&nbsp;&nbsp;&nbsp;&emsp;└── ... <br />

The model will be saved in `$SAVE_DIR$/$MODEL_NAME$`.

### Train stage 2
In the second training stage, you can train the model on the synthetic dataset and the clinical dataset with adverserial learning:
```
python train_stage_2.py \
    --model_name=$MODEL_NAME$ \
    --com_dataset_dir=$COM_DATASET_DIR$ \
    --flaw_dataset_dir=$FLAW_DATASET_DIR$ \
    --save_dir=$SAVE_DIR$ \
    --log_interval=1 --sample_interval=1 \
    --save_interval=2 --batch_size=1
```
`$COM_DATASET_DIR$` and `$FLAW_DATASET_DIR$` are the paths to the synthetic dataset and clinical dataset respectively. The structure of each dataset directory is the same to the structure shown in train stage 1.

### Train unet module
Use the trained CBCT restoration model to generate the restored images in `$DATASET_DIR$/train/rec_image` with the following command:
```
python train_unet_module.py \
    --dataset_dir=$COM_DATASET_DIR$ \
    --inp_model_path=$INP_MODEL_PATH$ \
    --prepare_image_flag
```
`$INP_MODEL_PATH$` is the path of model weight trained in stage 2. 

Generate corresponding predicted label in `$DATASET_DIR$/train/rec_label` using boolean operation on the maxillary segmentations of `image` and `rec_image`.

Construct hdf5 dataset and train the cleft defect mask prediction module:
```
python train_unet_module.py \
    --model_name=$UNET_MODULE_NAME$ \
    --dataset_dir=$COM_DATASET_DIR$ \
    --inp_model_path=$INP_MODEL_PATH$ \
    --save_dir=$SAVE_DIR$ \
    --prepare_hdf5_flag --train_unet_flag
```
The weight of the cleft defect mask prediction module will be saved in `$SAVE_DIR$/$UNET_MODULE_NAME$`.

## Citation
If you find this repository useful, please cite: <br /> 
**Stochastic Anomaly Simulation for Maxilla Completion from Cone-Beam Computed Tomography** <br />
Yixiao Guo, Yuru Pei, Si Chen, Zhibo Zhou, Tianmin Xu, Hongbin Zha. <br />
Medical Image Computing and Computer Assisted Intervention (MICCAI), 2024. 