import concurrent.futures
# import multiprocessing
from rqalpha import run_file

tasks = []
for short_period in range(3, 10, 2):
    for long_period in range(30, 90, 5):
        config = {
            "extra": {
                "context_vars": {
                    "SHORTPERIOD": short_period,
                    "LONGPERIOD": long_period,
                },
                "log_level": "error",
            },
            "base": {
                "data_bundle_path": "D:\\rqalpha\data_bundle\\bundle",
                "start_date": "2013-01-01",
                "end_date": "2019-06-15",
                "benchmark": "000300.XSHG",
                "frequency": "1d",
                "strategy_file": ".py",
        "accounts": {
            "stock": 1000000
        }
            },
            "mod": {
                "sys_progress": {
                    "enabled": True,
                    "show": True,
                },
                "sys_analyser": {
                    "enabled": True,
                    "output_file": "out-{short_period}-{long_period}.pkl".format(
                        short_period=short_period,
                        long_period=long_period,
                    )
                },
            },
        }

        tasks.append(config)


def run_bt(config):
    file_path = "C:\\Users\wilsonZhang\PycharmProjects\quant_ml\\rqalpha_backtest\Tune_parameter\Golden_cross.py"
    run_file(file_path, config)

#
# for task in tasks:
#     run_bt(task)
#
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    for task in tasks:
        executor.submit(run_bt, task)




