import os
import time
import pickle

import hpbandster.distributed.utils

from hpbandster.config_generators.bohb import BOHB
from hpbandster.config_generators.kde_ei import KDEEI as tpe
from hpbandster.config_generators import RandomSampling
from hpbandster.HB_iteration import SuccessiveResampling, SuccessiveHalving


def standard_parser_args(parser):
    parser.add_argument('--dest_dir', type=str,
                        help='the destination directory. A new subfolder is created for each benchmark/dataset.',
                        default='../data/')
    parser.add_argument('--num_iterations', type=int, help='number of Hyperband iterations performed.', default=4)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--method', type=str, default='randomsearch')
    parser.add_argument('--num_local_workers', type=int, help='how many local workers are spawned', default=1)
    parser.add_argument('--run_data_dir', type=str, help='directory where to store the live rundata', default=None)

    return (parser)


class MasterNode(object):
    def __init__(self, parsed_args, worker_class, subdir_postfix, host='localhost'):
        self.args = parsed_args
        self.host = host

        self.local_workers = []
        self.worker_class = worker_class

        self.run_id = worker_class.data_subdir(parsed_args) + str(self.args.run_id)
        self.subdir = os.path.join(parsed_args.dest_dir, worker_class.data_subdir(parsed_args), subdir_postfix)
        os.makedirs(self.subdir, exist_ok=True)

        self.min_budget, self.max_budget = self.worker_class.get_budgets(self.args)

    def args_to_optimitzer(self):

        CS = self.worker_class.get_config_space(self.args)
        SH_iteration = SuccessiveHalving

        self.eta = 3
        CG = None

        if self.args.method == 'hyperband2':
            CG = RandomSampling(CS, directory=self.args.run_data_dir)
            self.eta = 2

        if self.args.method == 'hyperband3':
            CG = RandomSampling(CS, directory=self.args.run_data_dir)

        if self.args.method == 'hyperband9':
            CG = RandomSampling(CS, directory=self.args.run_data_dir)
            self.eta = 9

        if self.args.method == 'randomsearch':
            CG = RandomSampling(CS, directory=self.args.run_data_dir)
            self.min_budget = self.max_budget

        if self.args.method == 'BO-HB-bayesopt':
            CG = tpe(CS, top_n_percent=15, min_points_in_model=2 * len(CS.get_hyperparameters()),
                     update_after_n_points=1, num_samples=256, mode='sampling', random_fraction=1. / 3.,
                     directory=self.args.run_data_dir, overwrite=False)

        if self.args.method == 'BO-HB-CIFAR':
            CG = BOHB(CS, directory=self.args.run_data_dir, random_fraction=0.1, num_samples=256)

        if self.args.method == 'BO-HB3':
            CG = BOHB(CS, directory=self.args.run_data_dir)

        if self.args.method == 'BO-HB4':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 4
        if self.args.method == 'BO-HB5':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 5
        if self.args.method == 'BO-HB6':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 6
        if self.args.method == 'BO-HB7':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 7
        if self.args.method == 'BO-HB8':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 8
        if self.args.method == 'BO-HB9':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.eta = 9

        if self.args.method == 'BO-HB-1bw':
            CG = BOHB(CS, bandwidth_factor=1)
        if self.args.method == 'BO-HB-2bw':
            CG = BOHB(CS, bandwidth_factor=2)
        if self.args.method == 'BO-HB-3bw':
            CG = BOHB(CS, bandwidth_factor=3)
        if self.args.method == 'BO-HB-4bw':
            CG = BOHB(CS, bandwidth_factor=4)

        if self.args.method == 'BO-HB-1s':
            CG = BOHB(CS, num_samples=1)
        if self.args.method == 'BO-HB-4s':
            CG = BOHB(CS, num_samples=4)
        if self.args.method == 'BO-HB-8s':
            CG = BOHB(CS, num_samples=8)
        if self.args.method == 'BO-HB-16s':
            CG = BOHB(CS, num_samples=16)
        if self.args.method == 'BO-HB-64s':
            CG = BOHB(CS, num_samples=64)
        if self.args.method == 'BO-HB-256s':
            CG = BOHB(CS, num_samples=256)
        if self.args.method == 'BO-HB-1024s':
            CG = BOHB(CS, num_samples=1024)

        if self.args.method == 'BO-HB-0r':
            CG = BOHB(CS, random_fraction=0)
        if self.args.method == 'BO-HB-10r':
            CG = BOHB(CS, random_fraction=0.1)
        if self.args.method == 'BO-HB-20r':
            CG = BOHB(CS, random_fraction=0.2)
        if self.args.method == 'BO-HB-30r':
            CG = BOHB(CS, random_fraction=0.3)
        if self.args.method == 'BO-HB-40r':
            CG = BOHB(CS, random_fraction=0.4)
        if self.args.method == 'BO-HB-50r':
            CG = BOHB(CS, random_fraction=0.5)
        if self.args.method == 'BO-HB-60r':
            CG = BOHB(CS, random_fraction=0.6)
        if self.args.method == 'BO-HB-70r':
            CG = BOHB(CS, random_fraction=0.7)
        if self.args.method == 'BO-HB-80r':
            CG = BOHB(CS, random_fraction=0.8)
        if self.args.method == 'BO-HB-90r':
            CG = BOHB(CS, random_fraction=0.9)

        if self.args.method == 'my_tpe':
            CG = BOHB(CS, directory=self.args.run_data_dir)
            self.min_budget = self.max_budget

        if CG is None:
            raise ValueError('--method option %s not known!' % self.args.method)

        return (CG, SH_iteration)

    def pyro_conf_filename(self):
        return (os.path.abspath(os.path.join(self.subdir, 'pyro_conf_%i.pkl' % self.args.run_id)))

    def start_name_server(self):
        """ starts nameserver and provides information to other nodes"""

        self.ns_host, self.ns_port = hpbandster.distributed.utils.start_local_nameserver(host=self.host, port=0)
        with open(self.pyro_conf_filename(), 'wb') as fh:
            pickle.dump((self.ns_host, self.ns_port), fh)

    def find_name_server(self, num_tries=60, interval=1):
        for i in range(num_tries):
            try:
                with open(self.pyro_conf_filename(), 'rb') as fh:
                    self.ns_host, self.ns_port = pickle.load(fh)
                return
            except FileNotFoundError:
                print('config file %s not found (trail %i/%i)' % (self.pyro_conf_filename(), i + 1, num_tries))
                time.sleep(interval)
            except:
                raise

        raise RuntimeError("Could not find the nameserver information, aborting!")

    def start_local_workers(self, background=True):

        """ For surrogates we can use local workers"""
        while len(self.local_workers) < self.args.num_local_workers:
            w = self.worker_class(self.args, nameserver=self.ns_host, ns_port=self.ns_port, run_id=self.run_id,
                                  id=len(self.local_workers), host=self.host)
            w.run(background=background)
            self.local_workers.append(w)

    def clean_up(self):
        """ deletes all unneccessary files in the end """
        os.remove(self.pyro_conf_filename())

    def run(self, ConfigGenerator=None, SH_iteration=None, ping_interval=60, queue_sizes=None, min_n_workers=1,
            dynamic_queue_size=False, save_to_pickle=True):
        """ actually running the chosen method on the chosen benchmark """

        if queue_sizes is None:
            queue_sizes = (min_n_workers - 1, min_n_workers)

        CG, SH_iter = self.args_to_optimitzer()

        CG = CG if ConfigGenerator is None else ConfigGenerator
        SH_iter = SH_iter if SH_iteration is None else SH_iteration

        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            eta=self.eta, min_budget=self.min_budget,  # HB parameters
            max_budget=self.max_budget,
            run_id=self.run_id,
            working_directory=self.subdir,
            nameserver=self.ns_host,
            ns_port=self.ns_port,
            host=self.host,
            ping_interval=ping_interval,
            job_queue_sizes=queue_sizes,
            dynamic_queue_size=dynamic_queue_size,
        )

        res = HB.run(self.args.num_iterations, iteration_class=SH_iter, min_n_workers=min_n_workers)
        HB.shutdown(shutdown_workers=True)

        if save_to_pickle:
            with open(os.path.join(self.subdir, '%s_run_%i.pkl' % (self.args.method, self.args.run_id)), 'wb') as fh:
                pickle.dump(res, fh)

        return (res)
