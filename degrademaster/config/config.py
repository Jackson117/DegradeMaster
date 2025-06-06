import argparse
import yaml

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="PROTAC Degradation Prediction")

    # General settings
    parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test'], help='Run mode')
    parser.add_argument('--dataset', type=str, help='Dataset JSON name (without extension)')
    parser.add_argument('--data_format', type=str, default='csv', help='The format of input data list')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_rate', type=float, default=0., help='Train/val split ratio')
    parser.add_argument('--show_input', type=bool, default=True, help='Show input')

    # Model architecture
    parser.add_argument('--conv_name', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE', 'EGNN'], help='Type of graph convolution layer')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers (used in EGNN)')
    parser.add_argument('--attention', action='store_true', help='Use attention in EGNN')

    # Feature setup
    parser.add_argument('--feature', action='store_true', help='Use real features instead of random')
    parser.add_argument('--select_pocket_war', type=int, default=5, help='Pocket warhead selection threshold')
    parser.add_argument('--select_pocket_e3', type=int, default=5, help='Pocket E3 selection threshold')

    # Graph dimension settings
    parser.add_argument('--e3_dim', type=int, default=30, help='Ligase pocket graph feature dimension')
    parser.add_argument('--protac_dim', type=int, default=167, help='PROTAC graph feature dimension')
    parser.add_argument('--tar_dim', type=int, default=30, help='Target pocket graph feature dimension')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')

    return parser.parse_args()
