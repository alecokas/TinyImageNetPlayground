from argparse import ArgumentParser
from datetime import datetime
from keras.callbacks import CSVLogger, ModelCheckpoint
import os


def create_output_directory(output_directory):
    """ Creates a directory for the current training session based on the date and time.
        Log files and the model are saved here.
    """
    dir_name = output_directory + str(datetime_as_string())
    os.makedirs(dir_name)
    print("Created output directory: {}".format(dir_name))
    return dir_name


def datetime_as_string():
    """ Generate the date and time as a string so that it can be used in the file name.
    """
    date_time = get_current_datetime()
    datetime_as_string = "{:04}-{:02}-{:02}_{:02}.{:02}.{:02}".format(date_time[0], date_time[1], date_time[2],
                                                                      date_time[3], date_time[4], date_time[5])
    return datetime_as_string


def get_current_datetime():
    """ Gets six values corresponding to the current date and time (year, month, day, hour, min, second)
    """
    date_time = datetime.today()
    return date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second


def save_model(model, model_num, output_directory_name):
    """ Save the model architecture in a JSON file.
    """
    model_json = model.to_json()
    with open(output_directory_name + "/model" + str(model_num) + ".json", "w") as json_file:
        json_file.write(model_json)


def setup_callbacks(output_directory_name):
    """
    """
    callbacks = []
    # Log learning curve data
    csv_logger_callback = CSVLogger(output_directory_name + "/log.csv", append=True, separator=',')
    callbacks.append(csv_logger_callback)

    # Save model weights after each epoch
    model_checkpoint_callback = ModelCheckpoint(
        filepath=output_directory_name + "/" + "weights_{epoch:02d}.h5",
        save_weights_only=True,
        period=1
    )
    callbacks.append(model_checkpoint_callback)
    return callbacks


def parse_args(args):
    """ Parse commandline arguments to the train_models.py script.
    """
    # Ignore the script file name stored at args[0]
    args = args[1:]

    parser = ArgumentParser(description='Utility function for parsing commandline arguments to the train_models script')
    parser.add_argument(
        "-d",
        "--data_location",
        help="The weights will be loaded from a file located in the argument model_weights_filepath",
        type=str
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        help="The location of model weights, logs, and the model architecture.",
        nargs="?",
        type=str
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        help="Number of epochs",
        nargs="?",
        default="10",
        type=int
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of batches per gradient descent update",
        nargs="?",
        default="64",
        type=int
    )

    parsed_args = parser.parse_args(args)
    if not parsed_args.data_location:
        raise RuntimeError("Please input a folder location for the dataset")
    if not os.path.isdir(parsed_args.data_location):
            raise RuntimeError("The data_location folder does not exist")
    if parsed_args.output_directory:
        if not os.path.isdir(parsed_args.output_directory):
            raise RuntimeError("The output_directory folder does not exist")

    return parsed_args
