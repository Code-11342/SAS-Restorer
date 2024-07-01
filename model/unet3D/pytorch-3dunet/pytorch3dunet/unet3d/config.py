import os
import argparse
from email.policy import default

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--train', action="store_true", help='(Added) Whether is train')
    parser.add_argument('--train_file_paths', type=str, help='(Added) paths of train file', default=None)
    parser.add_argument('--val_file_paths', type=str, help='(Added) paths of val file', default=None)
    parser.add_argument('--test_file_paths', type=str, help='(Added) paths of test file', default=None)
    parser.add_argument('--checkpoint_dir', type=str, help='(Added) dir of checkpoint, for train stage', default=None)
    parser.add_argument('--checkpoint_path', type=str, help='(Added) path of checkpoint, for test stage', default=None)
    parser.add_argument('--output_dir', type=str, help='(Added) path of output', default=None)
    # single classification or multiple classification
    parser.add_argument("--single_category", action="store_true")
    args = parser.parse_args()
    print(args)
    config = _load_config_yaml(args.config)
    
    # Added modify config
    # Dataset
    if(args.train):
        # train
        print(f"\n[Adjust Dataset] Modify train_file_paths from {config['loaders']['train']['file_paths']} to {[args.train_file_paths]}\n")
        config["loaders"]["train"]["file_paths"] = [args.train_file_paths]
        # val
        print(f"\n[Adjust Dataset] Modify val_file_paths from {config['loaders']['val']['file_paths']} to {[args.val_file_paths]}\n")
        config["loaders"]["val"]["file_paths"] = [args.val_file_paths]
    else:
        # test
        print(f"\n[Adjust Dataset] Modify test_file_paths from {config['loaders']['test']['file_paths']} to {[args.test_file_paths]}\n")
        config["loaders"]["test"]["file_paths"] = [args.test_file_paths]

    # Checkpoint
    if(args.train):
        # train
        print(f"\n[Adjust Checkpoint] Modify checkpoint_path from {config['trainer']['checkpoint_dir']} to {args.checkpoint_dir}\n")
        config["trainer"]["checkpoint_dir"] = args.checkpoint_dir
    else:
        # test
        best_checkpoint_path = f"{args.checkpoint_path}"
        os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
        print(f"\n[Adjust Checkpoint] Modify checkpoint_path from {config['model_path']} to {best_checkpoint_path}\n")
        config["model_path"] = best_checkpoint_path
    
    # Output
    if(not args.train):
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        print(f"\n[Adjust Output] Modify output_dir from {config['loaders']['output_dir']} to {args.output_dir}\n")
        config['loaders']["output_dir"] = args.output_dir
    
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
