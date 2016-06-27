import os, configparser
from Util.Params.pyVSParams import config_dict


class pyVoxelStats:
    def __init__(self):
        self.config = self.check_config()

    def check_config(self, config_path='~/.pyVS/pyVSParams.ini'):
        if os._exists(config_path):
            cf = configparser.ConfigParser()
            cf.read(config_path)
            return cf
        else:
            os.makedirs(os.path.dirname(config_path))
            cf = configparser.ConfigParser()
            for k, v in config_dict:
                cf[k] = v
            with open(config_path, 'w') as configfile:
                cf.write(configfile)
