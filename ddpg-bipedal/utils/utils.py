import yaml
import os
import sys

def default_path(filename):
    return os.path.join(sys.path[0], filename)

# load arguments from args.yaml
def load_args(filename='args.yaml'):
    with open(default_path(filename), 'r') as f:
        try:
            yaml_f = yaml.load(f)
            return yaml_f
        except yaml.YAMLError as exc:
            print(exc)

# save args to args.yaml
def save_args(args, args_to_update=None, filename='args.yaml'):
    if args_to_update is None:
        args_to_update = load_args(filename)

    with open(default_path(filename), 'w') as f:
        try:
            args_to_update.update(args)
            yaml.dump(args_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)
    