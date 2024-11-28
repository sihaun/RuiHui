import argparse
import json
import os

def load_config(config_file='config.json'):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found!")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def config():
    """Configuration settings."""
    # Load configuration from JSON file
    default_config = load_config()

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments with default values from JSON
    parser.add_argument('-a', '--arch', metavar='ARCH', default=default_config['arch'],
                        choices=['tf_efficientnet_b7'],
                        help='model architecture: ' +
                             ' | '.join(['vgg', 'tf_efficientnet_b7']) +
                             f' (default: {default_config["arch"]})')
    parser.add_argument('--datapath', default=default_config['datapath'], type=str, metavar='PATH',
                        help=f'where you want to load/save your dataset? (default: {default_config["datapath"]})')
    parser.add_argument('--classes', nargs='+', default=default_config['classes'],
                        type=str, metavar='CLASS',
                        help='List of classes for classification (default: aibao fubao huibao lebao ruibao)')    
    parser.add_argument('--epochs', default=default_config['epochs'], type=int, metavar='N',
                        help=f'number of total epochs to run (default: {default_config["epochs"]})')
    parser.add_argument('-b', '--batch-size', default=default_config['batch_size'], type=int, metavar='N',
                        help=f'mini-batch size (default: {default_config["batch_size"]})')
    parser.add_argument('--lr', '--learning-rate', default=default_config['lr'], type=float,
                        metavar='LR', help=f'initial learning rate (default: {default_config["lr"]})',
                        dest='lr')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true', default=default_config['cuda'],
                        help='use cuda?')
    parser.add_argument('--save', default=default_config['save'], type=str, metavar='FILE.pth',
                        help=f'name of checkpoint for saving model (default: {default_config["save"]})')

    # Parse command-line arguments
    cfg = parser.parse_args()

    return cfg
