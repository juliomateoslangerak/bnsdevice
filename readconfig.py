# import ConfigParser
import os
from configparser import ConfigParser

path = os.path.dirname(os.path.abspath(__file__))
files = [os.path.sep.join([path, file])
    for file in os.listdir(path) if file.endswith('.conf')]
config = ConfigParser()
config.read(files)