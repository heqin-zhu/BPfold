__all__ = ["get_config", "update_config", "getEasyDict", "read_yaml", "toYaml", "write_yaml"]

import yaml


def getEasyDict(dic):
    new_dic = dic
    for k, v in dic.items():
        if isinstance(v, dict):
            new_dic[k] = getEasyDict(v)
        else:
            new_dic[k] = v
    return EasyDict(new_dic)


class EasyDict(object):
    def __init__(self, opt):
        self.opt = opt

    def __getattribute__(self, name):
        if name == 'opt' or name.startswith("_") or name not in self.opt:
            return object.__getattribute__(self, name)
        else:
            return self.opt[name]

    def __setattr__(self, name, value):
        if name == 'opt':
            object.__setattr__(self, name, value)
        else:
            self.opt[name] = value

    def __getitem__(self, name):
        return self.opt[name]

    def __setitem__(self, name, value):
        self.opt[name] = value

    def __contains__(self, item):
        return item in self.opt

    def __repr__(self):
        return self.opt.__repr__()

    def __len__(self):
        return len(self.opt)

    def keys(self):
        return self.opt.keys()

    def values(self):
        return self.opt.values()

    def items(self):
        return self.opt.items()

    def update(self, dic):
        for k, v in dic.items():
            self.opt[k] = v


def resolve_expression(config):
    if type(config) is dict:
        new_config = {}
        for k, v in config.items():
            if type(v) is str and v.startswith("!!python"):
                v = eval(v[8:])
            elif type(v) is dict:
                v = resolve_expression(v)
            new_config[k] = v
        config = new_config
    return config


def get_config(config_file, config_names=[]):
    ''' load config from file
    '''
    config = resolve_expression(read_yaml(config_file))
    if type(config_names) == str:
        return getEasyDict(config[config_names])
    while len(config_names) != 0:
        config_name = config_names.pop(0)
        if config_name not in config:
            raise ValueError("Invalid config name: {}".format(config_name))
        config = config[config_name]

    return getEasyDict(config)


def update_config(config, args):
    ''' rewrite default config with user input
    '''
    if args is None:
        return
    if hasattr(args, "__dict__"):
        args = args.__dict__
    for arg, val in args.items():
        # if not (val is None or val is False) and arg in config: config[arg] = val
        # TO FIX: this may cause bugs for other programs
        if arg in config and val is not None:
            config[arg] = val

    for _, val in config.items():
        if isinstance(val, dict) or isinstance(val, EasyDict):
            update_config(val, args)


def resolve_obj(obj):
    if isinstance(obj, list):
        return [resolve_obj(i) for i in obj]
    elif isinstance(obj, dict) or isinstance(obj, EasyDict):
        return {k: resolve_obj(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def read_yaml(path):
    with open(path) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def toYaml(path, dic, listOneLine=True):
    dic = resolve_obj(dic)
    if listOneLine:
        data_str = yaml.dump(dic)
    else:
        data_str = yaml.dump(dic, default_flow_style=False)
    with open(path, 'w') as f:
        f.write(data_str)
write_yaml = toYaml
