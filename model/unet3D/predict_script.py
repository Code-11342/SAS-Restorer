# Notes:
# 此处替换config文件的参数有:
# 1. train, val, test 的输入数据路径, 改为 train_file_paths/val_file_paths/test_file_paths
# 2. train 和 test 的模型路径, train将训练模型的checkpoint_dir改为输入的checkpoint_dir, 
#                             test将测试模型的model_path改为checkpoint_dir下的best_checkpoint.pytorch)
# 3. test 的输出数据路径, 将output_dir改为输入的output_dir
import sys
sys.path.append("../../")
from dependencies import *
from utils import *
import h5py
from argparse import ArgumentParser
arg_parser = ArgumentParser()
arg_parser.add_argument("--device", type=str, default="0")
# model config
arg_parser.add_argument("--checkpoint_path", type=str, default=None)
arg_parser.add_argument("--config_path", type=str, default="./default_config.yaml")
# process config
arg_parser.add_argument("--clean", action="store_true")
arg_parser.add_argument("--prepare", action="store_true")
arg_parser.add_argument("--infer", action="store_true")
arg_parser.add_argument("--convert_output", action="store_true")
# path config
# eval data dir
arg_parser.add_argument("--input_image_dir", type=str)
arg_parser.add_argument("--restore_image_dir", type=str)
arg_parser.add_argument("--output_label_dir", type=str)
args = arg_parser.parse_args()

# model config
# config path
config_path = os.path.abspath(args.config_path)
# checkpoint path
checkpoint_path = os.path.abspath(args.checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

# path config
# eval data dir
input_image_dir = args.input_image_dir
restore_image_dir = args.restore_image_dir
output_label_dir = args.output_label_dir
# model data dir
model_data_dir = f"{checkpoint_dir}/model_data_dir"
model_input_dir = os.path.abspath(f"{model_data_dir}/1_model_input")
model_output_dir = os.path.abspath(f"{model_data_dir}/2_model_output")
# create dir
os.makedirs(model_input_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# clean model input and output
if(args.clean):
    print("\n[Clean]")
    os.system(f"rm {output_label_dir}/*")
    os.system(f"rm {model_input_dir}/*")
    os.system(f"rm {model_output_dir}/*")

# copy and cvt input data
if(args.prepare):
    print("\n[Prepare]")
    print("Copying and Converting data")
    def hdf5_gen_func(input_image_path):
        name = get_name(input_image_path)
        restore_image_path = f"{restore_image_dir}/{name}.nii.gz"
        print(f"handling name: {name}")

        # load input image
        input_image = read_mha_array3D(input_image_path)
        # load restore image
        restore_image = read_mha_array3D(restore_image_path)
        # cvt hdf5
        dst_hdf5_path = f"{model_input_dir}/{name}.h5"
        hdf5_file = h5py.File(dst_hdf5_path, mode="w")
        input_data = np.stack([input_image, restore_image])
        hdf5_file.create_dataset(name="raw", shape=input_data.shape, dtype=input_data.dtype, data=input_data)
        hdf5_file.close()
        return True

    input_image_paths = glob.glob(f"{input_image_dir}/*nii.gz")
    multi_process_exec(process_func=hdf5_gen_func, process_num=10, data_list=input_image_paths)

# model inference
if(args.infer):
    print("\n[Infer]")
    device = args.device
    config_path = config_path
    test_file_paths = model_input_dir
    output_dir = model_output_dir
    os.chdir("./pytorch-3dunet")
    sys_cmd = f"CUDA_VISIBLE_DEVICES={device} python pytorch3dunet/predict.py"
    sys_cmd += f" --config={config_path}"
    sys_cmd += f" --checkpoint_path={checkpoint_path}"
    sys_cmd += f" --test_file_paths={test_file_paths}"
    sys_cmd += f" --output_dir={output_dir}"
    print(f"[Command]:\n{sys_cmd}\n")
    os.system(sys_cmd)
    os.chdir("../")

# cvt output data
if(args.convert_output):
    print("\n[Convert Output]")
    for output_hdf5_path in tqdm(glob.glob(f"{model_output_dir}/*")):
        name = get_name(output_hdf5_path).replace("_predictions","")
        output_label_path = f"{output_label_dir}/{name}.nii.gz"
        hdf5_file = h5py.File(output_hdf5_path)
        predict = hdf5_file["predictions"][...].astype(np.float32)
        print(f"test hdf5 prediction.shape:{predict.shape} type:{type(predict)} sum:{np.sum(predict)}")
        make_parent_dir(output_label_path)
        write_mha_array3D(predict, output_label_path)
        print(f"test saving cvt output_label_path:{output_label_path}")
