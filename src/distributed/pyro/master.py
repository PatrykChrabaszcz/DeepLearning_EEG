from hpbandster.distributed.dispatcher import Dispatcher


class Master:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--name_server", type=str,
                            help="")
        parser.add_argument("--ns_port", type=int,
                            help="")
        parser.add_argument("--run_id", type=int,
                            help="")
        parser.add_argument("--ping_interval", type=int,
                            help="")
        parser.add_argument("--host", type=int,
                            help="")

    def __init__(self, result_callback, queue_callback, name_server, ns_port, run_id, host, ping_interval):
        self.dispatcher = Dispatcher(result_callback, run_id, ping_interval,
                                     name_server, ns_port, host, queue_callback)

        def submit_job(self, job):
            self.dispatcher.submit_job(job.id)

        def run(self, thread=True):
            pass