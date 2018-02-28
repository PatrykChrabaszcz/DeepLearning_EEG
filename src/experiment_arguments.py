from ConfigSpace.read_and_write.pcs_new import read as read_pcs
from configparser import ConfigParser
from argparse import ArgumentParser
from copy import deepcopy
import logging
import json
import os
import io


# Initialize logging
logger = logging.getLogger(__name__)


class ExperimentArguments(object):

    # We want to extend standard ArgumentParser with sections
    # Maybe there is a better approach for that
    class ExperimentArgumentsParser:
        def __init__(self, use_all_cli_args=True):
            self.use_all_cli_args = use_all_cli_args
            self._parser = ArgumentParser(allow_abbrev=False)
            self.current_section = None

            # To find section based on argument name
            self._name_to_section = {}
            # Dictionary with section as first key and argument name as second key
            self._args = {}

        def section(self, section):
            self.current_section = section
            if section not in self._args.keys():
                self._args[section] = {}

        def add_argument(self, name, **kwargs):
            assert self.current_section is not None, 'You need to specify arguments section'
            assert name not in self._name_to_section.keys(), 'Argument %s already exists' % name
            self._name_to_section[name] = self.current_section
            self._parser.add_argument('--%s' % name, **kwargs)

        def parse(self, unknown_args):
            args_dict = deepcopy(self._args)
            args, unknown_args = self._parser.parse_known_args(unknown_args)
            if self.use_all_cli_args:
                if len(unknown_args) > 0:
                    logger.error('Unknown CLI arguments %s' % unknown_args)
                    raise RuntimeError('Unknown CLI arguments %s' % unknown_args)

            # Create a dictionary based on arguments
            for arg, value in vars(args).items():
                section = self._name_to_section[arg]
                args_dict[section][arg] = value

            return args_dict

        def update_defaults(self, **kwargs):
            self._parser.set_defaults(**kwargs)

    def __init__(self, sections=None, use_all_cli_args=True):
        """
        Class used to extract arguments declared by different classes used for the run.
        Where possible default arguments are provided.
        Priority in which arguments are assigned, from the lowest:
        1. Default parameters specified in class declaration.
        2. Parameters provided by the user in the .ini file
        3. Parameters provided by the user as CLI arguments
        4. Parameters specified by Architecture Optimizer.

        :param sections: If not None then only specified .ini file sections will be used
        :param use_all_cli_args: If true then will assert that all CLI arguments were processed successfully.
        """

        self._sections = sections
        self._ini_conf = None
        self._ini_file_parser = ArgumentParser(allow_abbrev=False)
        self._ini_file_parser.add_argument("--ini_file", type=str, default="",
                                           help="Path to the file with default values "
                                                "for script parameters (.ini format).")

        self._parser = ExperimentArguments.ExperimentArgumentsParser(use_all_cli_args)
        self._arguments = None

    def save_to_file(self, file_path):
        """
        Save internal .ini file with all updates done to it (CLI arguments, ConfigSpace updates)
        to make it possible to restore experiment with parameters used for training.
        :param file_path: path to the output file
        :return
        """
        with open(file_path, 'w') as config_file:
            print(self._ini_conf)
            self._ini_conf.write(config_file)

    @staticmethod
    def read_configuration_space(file_path):
        """
        Load ConfigSpace object from .pcs (parameter configuration space) file.
        Throws an exception if file is not available.

        :param file_path: Path to the .pcs file.
        :return: ConfigSpace object representing configuration space.
        """

        with open(file_path, 'r') as f:
            s = f.readlines()
            config_space = read_pcs(s)
        return config_space

    def add_class_arguments(self, class_type):
        """
        Adds arguments specified in the class 'class_type' to the internal Argument Parser object.
        This function should be called sequentially for all classes used in the experiment.


        :return:
        """
        if self._arguments is not None:
            raise RuntimeError('Arguments already parsed!')

        self._parser.current_section = None
        class_type.add_arguments(self._parser)

    def get_arguments(self):
        """
        Gets a dictionary with parsed arguments.
        Argument source priority (highest first):
        1. CLI arguments provided by the user
        2. Arguments provided in the ini_file
        3. Defaults defined in the code


        :return: Dictionary with arguments

        """
        if self._arguments is not None:
            return self._arguments

        # Initialize ConfigParser to manage .ini file
        self._ini_conf = ConfigParser()

        # Get the file path for the ini_file
        args, unknown_args = self._ini_file_parser.parse_known_args()
        ini_file = args.ini_file

        # By default ini file is empty string which means that we don't use it
        if ini_file != "":
            logger.debug('Updating default parameter values from file: %s' % ini_file)
            self._ini_conf.read(args.ini_file)

            # Filtered sections if needed or all sections from the ini file

            args = self._parser.parse(unknown_args)

            sections = self._sections if self._sections is not None else args.keys()
            sections = set(sections).intersection(set(self._ini_conf.sections()))
            # Assert that arguments from .ini file are in the script
            for section in sections:
                ini_args_dict = dict(self._ini_conf.items(section))
                args_dict = args[section]

                for key in ini_args_dict.keys():
                    if key not in args_dict.keys():
                        raise RuntimeError('Argument %s: %s from .ini file not present '
                                           'in the script.' % (section, key))

                # Replace script defaults with defaults from the ini file
                self._parser.update_defaults(**ini_args_dict)

        # At this point we should have arguments from default values, .ini file values and CLI values
        args = self._parser.parse(unknown_args)

        # Now save everything to the ConfigParser, such that we will be able to save and restore those parameters
        # And flatten nested args such that we will be able to use them as **kwargs
        self._arguments = {}
        for section in args.keys():
            for arg_name, arg_value in args[section].items():
                if section not in self._ini_conf.sections():
                    self._ini_conf.add_section(section)
                self._ini_conf.set(section, arg_name, str(arg_value))
                self._arguments[arg_name] = arg_value

        return self._arguments

    def updated_with_configuration(self, configuration):
        """
        Will copy current parameters and update them based on configuration space.
        Configuration space can come from hyper-parameter optimizer.
        Does not modify original parameters!

        :param configuration: Configuration object (from ConfigSpace library)
        :return: Dictionary with updated arguments
        """

        arguments = self.copy()
        for arg_name in configuration.keys():
            new_arg_value = configuration[arg_name]
            old_arg_value = arguments[arg_name]
            logger.debug('Changing %s from %s to %s' % (arg_name, old_arg_value, new_arg_value))
            arguments[arg_name] = new_arg_value

        return arguments

    def copy(self):
        assert self._ini_conf is not None and self._arguments is not None, 'Can only copy initialized arguments!'

        experiment_arguments = ExperimentArguments()
        experiment_arguments._arguments = deepcopy(self._arguments)

        config_string = io.StringIO()
        self._ini_conf.write(config_string)
        # We must reset the buffer ready for reading.
        config_string.seek(0)
        experiment_arguments._ini_conf = ConfigParser()
        experiment_arguments._ini_conf.read_file(config_string)

        return experiment_arguments

    def __getattr__(self, item):
        if item[0] == '_':
            return super().__getattribute__(item)
        else:
            return self._arguments[item]

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        else:
            if key not in self._arguments.keys():
                raise KeyError('ExperimentArguments does not support addition of new arguments during runtime. '
                               '(Argument name: %s)' % key)
            self._arguments[key] = value

            # Makes it possible to save again updated by CLI arguments .ini
            # file and restore experiment
            for section in self._ini_conf.sections():
                if self._ini_conf.has_option(section, key):
                    self._ini_conf.set(section, key, str(value))
                    return
            raise RuntimeError('Could not set field %s in the ConfigParser object' % key)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)
