import sys
sys.path.append("../../")
from dependencies import *
from utils import *
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--model_name",type=str,default="default_seg")
arg_parser.add_argument("--device",type=str,default="0")
arg_parser.add_argument("--train_dataset_dir",type=str,default="default_data")
arg_parser.add_argument("--val_dataset_dir",type=str,default="default_data")
arg_parser.add_argument("--checkpoint_dir",type=str,default=None)
arg_parser.add_argument("--config_path",type=str,default="./default_config.yaml")
args = arg_parser.parse_args()

# paths
train_file_paths = os.path.abspath(f"{args.train_dataset_dir}")
val_file_paths = os.path.abspath(f"{args.val_dataset_dir}")
config_path = os.path.abspath(args.config_path)
make_parent_dir(config_path)
if(args.checkpoint_dir is None):
    checkpoint_dir = os.path.abspath(f"../../../save_dir/reference/{args.model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
else:
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

# configs
device = args.device
config_path = config_path
checkpoint_dir = checkpoint_dir
train_file_paths = train_file_paths
val_file_paths = val_file_paths

os.chdir("./pytorch-3dunet")
sys_cmd = f"CUDA_VISIBLE_DEVICES={device} python pytorch3dunet/train.py"
sys_cmd += f" --config={config_path}"
sys_cmd += f" --checkpoint_dir={checkpoint_dir}"
sys_cmd += f" --train_file_paths={train_file_paths}"
sys_cmd += f" --val_file_paths={val_file_paths}"
sys_cmd += f" --train"
print(f"[Command]:\n{sys_cmd}\n")
os.system(sys_cmd)
os.chdir("../")
