from base.env.pre_process_conf import active_stragery


def setup_market():
    pass

def setup_input_strategy():
    pass

def setup_algorithm():
    pass


class Automation:

    def __init__(self, setup_input_strategy, setup_market, setup_algorithm):
        self._input_strategy = setup_input_strategy()
        self._market = setup_market()
        self._algorithm = setup_algorithm()

    def fit(self):
        self._algorithm.run()
        return self._algorithm.eval_and_plot()

    def generate_input_strategy(self):
        pass

    def process(self):
        input_strategies = self._input_strategy()
        for idx, strategy in enumerate(input_strategies):
            active_stragery = ...
            df_output = self.fit()

        # generate back testing summary report



