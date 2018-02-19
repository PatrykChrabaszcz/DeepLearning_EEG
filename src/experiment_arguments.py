from argparse import ArgumentParser
from configparser import ConfigParser
from ConfigSpace.read_and_write.pcs_new import read as read_pcs
import logging
from copy import deepcopy

# Initialize logging
logger = logging.getLogger(__name__)


class ExperimentArguments(object):

    def __init__(self, use_all_cli_args=True):
        """
        Class used to extract arguments declared by different classes used for the run.
        Where possible default arguments are provided.
        Priority in which arguments are assigned, from the lowest:
        1. Default parameters specified in class declaration.
        2. Parameters provided by the user in the .ini file
        3. Parameters provided by the user as CLI arguments
        4. Parameters specified by Architecture Optimizer.

        :param use_all_cli_args: If true then will assert that all CLI arguments were processed successfully.
        """
        self._use_all_cli_args = use_all_cli_args
        self._parser = ArgumentParser()
        self._arguments = None

        self._ini_file_parser = ArgumentParser()
        self._ini_file_parser.add_argument("--ini_file", type=str, default="",
                                           help="Path to the file with default values "
                                                "for script parameters (.ini format).")

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

        :param class_type: Class Type used to extend argument list.
        :return:
        """
        if self._arguments is not None:
            raise RuntimeError('Can not add a new argument if all arguments are already parsed')
        class_type.add_arguments(self._parser)

    def get_arguments(self, sections=None, exclude_sections=None):
        """
        Gets a dictionary with parsed arguments.
        Argument source priority (highest first):
        1. CLI arguments provided by the user
        2. Arguments provided in the ini_file
        3. Defaults defined in the code

        :param sections: If not None then only specified .ini file sections will be used
        :param exclude_sections: If not None then will not process those .ini file sections

        :return: Dictionary with arguments

        """
        if self._arguments is not None:
            return self._arguments

        # Get the file path for the ini_file
        args, unknown_args = self._ini_file_parser.parse_known_args()

        # Static class variable to store unknown args. Helpful when we want to assert that all CLI arguments
        # are understood

        if args.ini_file != "":
            logger.debug('Updating default parameter values from file: %s' % args.ini_file)
            ini_conf = ConfigParser()
            ini_conf.read(args.ini_file)

            # Tell me if there is a better way to assert that .ini arguments are already in the parser
            args = self._parser.parse_known_args(unknown_args)[0]

            sections = sections if sections is not None else ini_conf.sections()
            exclude_sections = exclude_sections if exclude_sections is not None else []
            for section in sections:
                if section in exclude_sections:
                    continue

                # Replace script defaults with defaults from the ini file
                ini_args_dict = dict(ini_conf.items(section))
                for key in ini_args_dict:
                    if key not in vars(args).keys():
                        raise RuntimeError('Argument %s from .ini file not present in the script.' %
                                           key)
                self._parser.set_defaults(**ini_args_dict)

        args, unknown_args = self._parser.parse_known_args(unknown_args)

        # Good check to make sure that user did not make a mistake when providing experiment parameters.
        if self._use_all_cli_args:
            if len(unknown_args) > 0:
                logger.warning('There are some unknown arguments provided by the user')
                logger.error(unknown_args)
                raise RuntimeError('Unknown CLI arguments %s' % unknown_args)

        self._arguments = vars(args)
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
        experiment_arguments = ExperimentArguments()
        experiment_arguments._arguments = deepcopy(self._arguments)
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
            self._arguments[key] = value

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

