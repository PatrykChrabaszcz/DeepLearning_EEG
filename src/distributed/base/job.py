from ConfigSpace import ConfigurationSpace


class Job:
    def __init__(self, job_id, configuration):
        self.job_id = job_id
        self.configuration = configuration
