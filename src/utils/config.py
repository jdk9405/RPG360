import importlib
from yacs.config import CfgNode

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)


def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n
    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated
    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var

def load_class(filename, paths, concat=True):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.filename instead
    Parameters
    ----------
    filename : str
        Name of the file we are searching for
    paths : str or list of str
        Folders in which the file will be searched
    concat : bool
        Flag to concatenate filename to each path during the search
    Returns
    -------
    method : Function
        Loaded method
    """
    # for each path in paths
    for path in make_list(paths):
        # Create full path
        full_path = '{}.{}'.format(path, filename) if concat else path
        if importlib.util.find_spec(full_path):
            # Return method with same name as the file
            return getattr(importlib.import_module(full_path), filename)
    raise ValueError('Unknown class {}'.format(filename))


def get_default_config(cfg_default):
    config = load_class('get_cfg_defaults', 
                        paths=[cfg_default.replace('/', '.')],
                        concat=False)()
    config.merge_from_list(['default', cfg_default])
    return config

def parse_train_config(cfg_default, cfg_file):
    config = get_default_config(cfg_default)
    config = merge_cfg_file(config, cfg_file)
    #  return prepare_train_config(config)
    return config

def merge_cfg_file(config, cfg_file=None):
    """Merge configuration file"""
    if cfg_file is not None:
        config.merge_from_file(cfg_file)
        config.merge_from_list(['config', cfg_file])
    return config
