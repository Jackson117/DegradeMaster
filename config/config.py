import argparse
import yaml

def get_args(mode='name'):
    parser = argparse.ArgumentParser()
    if mode == 'name':
        parser.add_argument('--config', type=str, default='./config/config.yml')
    elif mode == 'case':
        parser.add_argument('--config', type=str, default='./config/config_c.yml')
    args = parser.parse_args()

    config = yaml.load(
        open(args.config),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**config)

    return args