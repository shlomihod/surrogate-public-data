class DPTask:

    def __init__(self):
        raise NotImplementedError

    def execute(self):
        raise

    @staticmethod
    def evaluate(self, results, **kwargs):
        raise NotImplementedError

    @staticmethod
    def plot(self, results, **kwargs):
        raise NotImplementedError
