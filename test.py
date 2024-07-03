from dependencies import *
from utils import *
from model import DilatedAutoEncoder

# cli
arg_parser = ArgumentParser()
# basic option
arg_parser.add_argument("--gpu_id", type=int, default=0)
# model option
arg_parser.add_argument("--inp_model_path", type=str, default="../save_dir/sas_model/newest_model.pth")
arg_parser.add_argument("--unet_module_path", type=str, default="../save_dir/sas_model/newest_model.pth")
# input image option
arg_parser.add_argument("--input_image_dir", type=str, default="../test_dir/image")
arg_parser.add_argument("--input_mask_dir", type=str, default="../test_dir/mask")
# output label option
arg_parser.add_argument("--output_image_dir", type=str, default="../test_dir/output_image_dir")
arg_parser.add_argument("--output_label_dir", type=str, default="../test_dir/output_label_dir")
# parse args
args = arg_parser.parse_args()

gpu_id = args.gpu_id
inp_model_path = args.inp_model_path
unet_module_path = args.unet_module_path
input_image_dir = args.input_image_dir
input_mask_dir = args.input_mask_dir
output_image_dir = args.output_image_dir
output_label_dir = args.output_label_dir

fill_value = 1300/2300
bone_thresh = 1500/2300
num_hiddens = 768
num_residual_hiddens = 256
num_residual_layers = 3
embedding_dim = 768
eps = 1/128**3

# init model
gen = DilatedAutoEncoder(in_channels=3,
                       num_hiddens=num_hiddens,
                       num_residual_layers=num_residual_layers,
                       num_residual_hiddens=num_residual_hiddens,
                       embedding_dim=embedding_dim)
assert(os.path.exists(inp_model_path))
gen.load_state_dict(torch.load(inp_model_path))
gen.eval()
gen = gen.cuda()

# predict reconstructed image
image_paths = glob.glob(f"{input_image_dir}/*nii.gz")
for image_path in tqdm(image_paths):
    name = get_name(image_path)
    print(f"infering {name}")
    mask_path = f"{input_mask_dir}/{name}.nii.gz"
    rec_image_path = f"{output_image_dir}/{name}.nii.gz"
    x = read_mha_tensor4D(image_path).unsqueeze(0)
    mask = read_mha_tensor4D(mask_path).unsqueeze(0)
    bony_border_mask = torch.where(x>bone_thresh, 1, 0).float()*mask
    x, mask, bony_border_mask = tocuda([x, mask, bony_border_mask])
    
    with torch.no_grad():
        masked_x = x*(1-mask) + fill_value*mask
        input_x = torch.cat([masked_x, mask, bony_border_mask], dim=1)
        rec_x = gen.forward(input_x)
        con_rec_x = x*(1-mask) + rec_x*mask
        write_mha_tensor4D(con_rec_x[0], rec_image_path)

# predict label with unet module
abs_cur_dir = get_abs_path("./")
abs_unet_dir = get_abs_path("./model/unet3D")
abs_input_image_dir = get_abs_path(input_image_dir)
abs_rec_image_dir = get_abs_path(output_image_dir)
abs_output_label_dir = get_abs_path(output_label_dir)
abs_checkpoint_path = get_abs_path(unet_module_path)
abs_config_path = get_abs_path("./model/unet3D/yamls/ours_unet_cleft/test_clp.yaml")
# run predict script
os.chdir(abs_unet_dir)
cmd = ""
cmd += f"CUDA_VISIBLE_DEVICES={gpu_id} python predict_script.py --device={gpu_id} "
cmd += f"--config_path={abs_config_path} "
cmd += f"--checkpoint_path={abs_checkpoint_path} "
cmd += f"--input_image_dir={abs_input_image_dir} "
cmd += f"--restore_image_dir={abs_rec_image_dir} "
cmd += f"--output_label_dir={abs_output_label_dir} "
cmd += f"--clean --prepare --infer --convert_output"
os.system(cmd)
os.chdir(abs_cur_dir)



