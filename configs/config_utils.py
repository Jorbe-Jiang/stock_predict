from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import tensorflow as tf


class Config(object):
    """Configuration"""

    def __init__(self):
      pass


def read_config_from_json_file(config_json):
    config = Config()
    with tf.gfile.GFile(config_json, "r") as reader:
        text = reader.read()
        json_obj = json.loads(text)
        for (key, value) in six.iteritems(json_obj):
            config.__dict__[key] = value

    return config


if __name__ == "__main__":
    config_json = "config.json"
    config = read_config_from_json_file(config_json)
    print(config)
    print(config.vocab_size)