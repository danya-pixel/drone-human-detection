from ultralytics.utils import yaml_load

if __name__ == "__main__":
    yaml = yaml_load("datasets/cfg/VisDrone.yaml")
    exec(yaml["download"])
