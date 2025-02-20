import json

class Arg:
  def __init__(self, **entries):
    self.__dict__.update(entries)

class ModelConfig:
  def __init__(self, config_path):
    self.config_path = config_path
    with open(self.config_path, 'r') as f:
      self.config_json = json.load(f)
    self.arg = Arg(**self.config_json)

  def get_config(self):
    return self.arg
  
class TrainConfig:
  def __init__(self, train_path):
    self.train_path = train_path
    with open(self.train_path, 'r') as f:
      self.train_json = json.load(f)
    self.arg = Arg(**self.train_json)

  def get_config(self):
    return self.arg
  