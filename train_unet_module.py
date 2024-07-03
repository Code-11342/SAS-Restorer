from dependencies import *
from utils import *
from model import DilatedAutoEncoder
import h5py

# cli
arg_parser = ArgumentParser()
# basic option
arg_parser.add_argument("--model_name", type=str, default="unet_module")
arg_parser.add_argument("--gpu_id", type=int, default=0)
# dataset option
arg_parser.add_argument("--dataset_dir", type=str, default="../dataset_dir/complete_dataset")
# save option
arg_parser.add_argument("--inp_model_path", type=str, default="../save_dir/inp_model/newest_model.pth")
arg_parser.add_argument("--save_dir", type=str, default="../save_dir")
# prepare unet data option
arg_parser.add_argument("--prepare_image_flag", action="store_true")
arg_parser.add_argument("--prepare_hdf5_flag", action="store_true")
# train stage option
arg_parser.add_argument("--train_unet_flag", action="store_true")
# parse args
args = arg_parser.parse_args()
prepare_image_flag = args.prepare_image_flag
prepare_hdf5_flag = args.prepare_hdf5_flag
train_unet_flag = args.train_unet_flag

# paths
model_name = args.model_name
gpu_id = args.gpu_id
dataset_dir = args.dataset_dir
save_dir = args.save_dir
model_path = args.model_path
model_dir = f"{save_dir}/{model_name}"
log_dir = f"{model_dir}/log_dir"

# dataset dir
# train
train_dataset_dir = f"{dataset_dir}/train"
# val
val_dataset_dir = f"{dataset_dir}/val"

# init log
log_path = f"{log_dir}/log.txt"
init_log(log_path)
log(args)

fill_value = 1300/2300
thresh_bone = 1500/2300
num_hiddens = 768
num_residual_hiddens = 256
num_residual_layers = 3
embedding_dim = 768
eps = 1/128**3

if(prepare_image_flag):
    log("[Prepare Data]")
    gen = DilatedAutoEncoder(in_channels=3,
                             num_hiddens=num_hiddens,
                             num_residual_layers=num_residual_layers,
                             num_residual_hiddens=num_residual_hiddens,
                             embedding_dim=embedding_dim)
    gen.load_state_dict(torch.load(model_path))
    gen = gen.cuda()
    gen.eval()
    
    # construct complete image using trained inpaint model
    for cur_dataset_dir in [train_dataset_dir, val_dataset_dir]:
        image_paths = glob.glob(f"{cur_dataset_dir}/image/*nii.gz")
        for image_path in tqdm(image_paths):
            name = get_name(image_path)
            mask_path = image_path.replace("image", "mask")
            rec_image_path = image_path.replace("image", "rec_image")
            hdf5_path = f"{cur_dataset_dir}/hdf5_data/{name}.h5"
            
            x = read_mha_tensor4D(image_path).unsqueeze(0)
            mask = read_mha_tensor4D(mask_path).unsqueeze(0)
            bony_border_mask = torch.where(x>thresh_bone, 1, 0).float()*mask
            x, mask, bony_border_mask = tocuda([x, mask, bony_border_mask])
            
            with torch.no_grad():
                masked_x = x*(1-mask) + fill_value*mask
                input_x = torch.cat([masked_x, mask, bony_border_mask], dim=1)
                rec_x = gen.forward(input_x)
                con_rec_x = x*(1-mask) + rec_x*mask
                write_mha_array4D(con_rec_x[0], rec_image_path)

if(prepare_hdf5_flag):
    log("[Prepare Hdf5]")
    for cur_dataset_dir in [train_dataset_dir, val_dataset_dir]:
        image_paths = glob.glob(f"{cur_dataset_dir}/image/*nii.gz")
        for image_path in tqdm(image_paths):
            name = get_name(image_path)
            rec_image_path = image_path.replace("image", "rec_image")
            rec_label_path = image_path.replace("image", "rec_label")
            hdf5_path = f"{cur_dataset_dir}/hdf5_data/{name}.h5"
            image = read_mha_array3D(image_path)
            rec_image = read_mha_array3D(rec_image_path)
            rec_label = read_mha_array3D(rec_label_path)
            input_x = np.stack([image, rec_image])
            make_parent_dir(hdf5_path)
            hdf5_file = h5py.File(hdf5_path, mode="w")
            hdf5_file.create_dataset(name="raw", shape=input_x.shape, dtype=input_x.dtype, data=input_x)
            hdf5_file.create_dataset(name="label", shape=rec_label.shape, dtype=rec_label.dtype, data=rec_label)
            hdf5_file.close()

if(train_unet_flag):
    log("[Train Unet]")
    abs_cur_dir = get_abs_path("./")
    abs_unet_dir = get_abs_path("./model/unet3D")
    abs_train_dataset_dir = get_abs_path(train_dataset_dir)
    abs_val_dataset_dir = get_abs_path(val_dataset_dir)
    abs_checkpoint_dir = get_abs_path(model_dir)
    abs_config_path = get_abs_path("./model/unet3D/yamls/ours_unet_cleft/train_clp.yaml")
    # run unet train script
    os.chdir(abs_unet_dir)
    cmd = ""
    cmd += f"CUDA_VISIBLE_DEVICES={gpu_id} python train_script.py --device={gpu_id} "
    cmd += f"--model_name={model_name} "
    cmd += f"--train_dataset_dir={abs_train_dataset_dir} "
    cmd += f"--val_dataset_dir={abs_val_dataset_dir} "
    cmd += f"--checkpoint_dir={abs_checkpoint_dir} "
    cmd += f"--config={abs_config_path} "
    os.system(cmd)
    os.chdir(abs_cur_dir)
    