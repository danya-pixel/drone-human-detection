from ultralytics.utils import yaml_load

yaml = yaml_load("datasets/cfg/VisDrone.yaml")

exec(yaml["download"])
