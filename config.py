import sys # for reading command line arguments
from configparser import ConfigParser
import torch
import numpy as np
from log import log
from dataclasses import dataclass
import json
import random

@dataclass
class DataType:
    """
    Represents datatype for torch/numpy arrays.
    """
    numpy: np.dtype
    torch: torch.dtype
    # Size in bytes
    size: int


def getFloatDtype(bits: int):
    if bits == 64:
        return DataType(np.float64, torch.float64, 8)
    elif bits == 32:
        return DataType(np.float32, torch.float32, 4)
    elif bits == 16:
        return DataType(np.float16, torch.float16, 2)
    else:
        raise ValueError(f"Cannot build a float dtype with {bits} bits")


#######################################################################
#                             LOAD CONFIG                             #
#######################################################################

def load_config(filename):
    config = ConfigParser()

    if not config.read(config_filename):
        log.error(f"Failed to read config file: {config_filename}")
        sys.exit(1)
    else:
        log.info(f"Read config file: {config_filename}")

    global PLATFORM_NAME, NUM_USERS, LOCAL_EPOCHS, GLOBAL_EPOCHS
    global PREPROCESSING_FRACTION, TRAINING_FRACTION
    PLATFORM_NAME = config["FL"]["platform"]
    NUM_USERS     = config["FL"].getint("num users")
    LOCAL_EPOCHS  = config["FL"].getint("local epochs")
    GLOBAL_EPOCHS = config["FL"].getint("global epochs")
    PREPROCESSING_FRACTION = config["FL"].getfloat("preprocessing fraction")
    TRAINING_FRACTION      = config["FL"].getfloat("training fraction")

    global LEARNING_RATE, MOMENTUM, BATCH_SIZE, TEST_BATCH_SIZE
    LEARNING_RATE   = config["ML"].getfloat("learning rate")
    MOMENTUM        = config["ML"].getfloat("momentum") 
    BATCH_SIZE      = config["ML"].getint("batch size")
    TEST_BATCH_SIZE = config["ML"].getint("test batch size")

    global INTERNAL_DTYPE, EXTERNAL_DTYPE
    INTERNAL_DTYPE  = getFloatDtype(config["DATATYPES"].getint("internal"))
    EXTERNAL_DTYPE  = getFloatDtype(config["DATATYPES"].getint("external"))

    global DATASET_FILENAME, VALIDATION_SIZE
    DATASET_FILENAME = config["INPUT"]["dataset path"]
    VALIDATION_SIZE  = config["INPUT"].getfloat("validation size")

    global MODEL_NAME, MODEL_ARGS
    MODEL_NAME = config["MODEL"]["name"]
    MODEL_ARGS = config["MODEL"]

    global EVAL_PER_EPOCH, RESULTS_FILE
    EVAL_PER_EPOCH = config["TESTING"].getboolean("evaluate per epoch")
    RESULTS_FILE   = config["TESTING"]["results file"]

    global RANDOM_SEED
    RANDOM_SEED = config["INPUT"].getint("random seed")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


config_filename = "config.ini" if len(sys.argv) < 2 else sys.argv[1]
load_config(config_filename)


def read_results():
    """
    Read the results JSON file from config
    """
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    return data


def add_results(obj):
    """
    Append the given serializable object to the results JSON file.
    """
    try:
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(obj)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent = 4)

