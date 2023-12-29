# spt-plugin
Template for SPT plugin development.

## Graph processing

### Development process

Addition of new graph processing plugins to SPT is done in three steps:
1. **Exploratory stage**: a user evaluates if their processing model is good or not by manually using the output and upload functions described in the prior section and examining their results locally or in an SPT instance after upload.
2. **Proposal stage**: If a user determines that their graph model is a good candidate for inclusion into SPT as a default graph deep learning plugin that should be run on every new study imported into SPT, they would containerize their model according to the Docker template defined for the external plugin system, upload it to a GitHub repo or equivalent, and open an issue or pull request for SPT to include their plugin.
3. **Inclusion stage**: The SPT maintainers would quality check the proposed plugin and, if found to be worth including, upload the container to the SPT Docker page and modify SPT code to pull down and use that Docker image by default in SPT.

### The plugin should look like this

Plugins are to be made available as Docker images, built from a Dockerfile following the template provided in `Dockerfile`.

A plugin should have the following commands available from anywhere in the Docker image:
* `spt-plugin-print-graph-request-configuration`, which prints to stdout the configuration file intended to be used by this plugin to fetch graphs from an SPT instance to use for model training. An empty configuration file and a shell script to do this is provided in this repo, as well as the command needed to make this available in the template `Dockerfile`.
* `spt-plugin-train-on-graphs` trains the model and outputs a CSV of importance scores that can be read by `spt graphs upload-importances`. A template `train.py` is provided that uses a command line interface specified in `train_cli.py`. The template `Dockerfile` provides a command to make this script available anywhere in the Docker image. Its arguments are
    1. `--input_directory`, the path to the directory containing the graphs to train on.
    2. `--config_file`, the path to the configuration file. This should be optional, and if not provided `spt-plugin-train-on-graphs` should use reasonable defaults.
    3. `--output_directory`, the path to the directory in which to save the trained model, importance scores, and any other artifacts deemed important enough to save, like performance reports.
* `spt-plugin-print-training-configuration`, which prints to stdout an example configuration file for running `spt-plugin-train-on-graphs`, populated either with example values or the reasonable defaults used by the command. An empty configuration file and a shell script to do this is provided in this repo, as well as the command needed to make this available in the template `Dockerfile`.
