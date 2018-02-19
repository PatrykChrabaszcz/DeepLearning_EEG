class Master(object):
    def __init__(self, result_callback):
        self.results_callback = result_callback

    def submit_job(self, job):
        pass

    def run(self):
        """
        Will start listening to the results in a separate thread. Whenever new results appears
        self.results_callback will be called
        :return:
        """
        raise NotImplementedError('Implement this method in your subclass')
