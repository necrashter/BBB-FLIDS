# BBB-FLIDS

This repository contains BBB-FLIDS, Basic Blockchain-Based Federated Learning Intrusion Detection System.

For detailed information about this project, please see [the report](report/main.pdf).


# Setup

You may want to see "Dummy Blockchain Platform" heading before following all instructions here.

## Virtual Env

Although it's not required, it's recommended to create a Python virtualenv for blockchain libraries.

First install virtualenv:
```sh
$ pip install virtualenv
```

Note that you might need elevated privileges:
```sh
$ sudo pip install virtualenv
```

After that, create the virtual environment in .venv folder:
```sh
# Create a virtual environment:
$ virtualenv -p python3 .venv
# Alternatively, enable the virtual environment to use system-wide packages:
# This allows you to avoid re-installing large libraries, e.g., pytorch
$ virtualenv -p python3 --system-site-packages .venv
```
It's recommended that you go with the latter option if you already have pytorch installed system-wide.


Activate your new virtual environment:
```sh
$ source .venv/bin/activate
```
Note that you will need to run this command each time you restart the terminal.

## Install Dependencies

Install the required dependencies from `requirements.txt`:
```sh
$ pip install -r requirements.txt
```

After that, you need to run `install_solc.py` file to install the Solidity compiler:
```sh
$ python install_solc.py
```

### Dummy Blockchain Platform

Thanks to `DummyPlatform` module, you can run this project without blockchain.
This is useful if:
- You don't want to install blockchain dependencies.
- Or you want to run the federated learning part only for better run-time performance.

Instead of installing all dependencies, install `numpy`, `pytorch`, `sklearn`, `pandas`, and `tqdm` (optional) only.
These are included in `fl-requirements.txt`:
```sh
$ pip install -r fl-requirements.txt
```

In `config.ini`, make sure that `platform` is set to `"dummy"`.

Now you should be able to run the project without blockchain.



# Running

This will run the simulation with the settings given in `config.ini`:
```sh
$ python main.py
```

You can supply your own ini file as an argument:
```sh
$ python main.py custom.ini
```

## Configuring

Most settings are set in `.ini` file.
Please see the comments in the given `config.ini` file for details.

In order to configure the neural network:
1. Create a new class in `ModelConfig.py`. Copy one of the examples for quickstart.
2. In `.ini` file, set the `model name` field to the class name of your new model.


## Plotting

If `evaluate per epoch` is enabled in configuration file, the model will be evaluated on the validation set after each global epoch (round).
Using the following command, you can plot the previous results you got:
```sh
$ python main.py
```

For this, `matplotlib` needs to be installed, which is included in neither `requirements.txt` nor `fl-requirements.txt`.
Thus, you need to install it manually:
```sh
$ pip install matplotlib
```

## Test Scripts

This will test some functionality in Solidity contract:
```sh
$ python TestContract.py
```

`StressTestContract.py` is intended to measure how much gas is required for a given byte-size of the machine-learning model.
```sh
$ python StressTestContract.py
```
