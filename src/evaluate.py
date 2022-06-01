import yaml

with open("./params.yaml") as conf_file:
    params = yaml.safe_load(conf_file)
