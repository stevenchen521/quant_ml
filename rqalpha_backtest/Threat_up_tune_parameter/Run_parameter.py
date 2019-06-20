import concurrent.futures
# import multiprocessing
from rqalpha import run_file

stock_list = ['000002.XSHE', '002475.XSHE', '601211.XSHG']
tasks = []
print(range(0, 1, 10))
for stock in stock_list:
    for order_percent in [x/10 for x in range(0, 10, 1)]:
        config = {
            "extra": {
                "context_vars": {
                    "s1": stock,
                    "ORDER_PERCENT": order_percent,
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
            "stock": 10000000
        }
            },
            "mod": {
                "sys_progress": {
                    "enabled": True,
                    "show": True,
                },
                "sys_analyser": {
                    "enabled": True,
                    "plot": False,
                    "output_file": "result\out-{stock}-{order_percent}.pkl".format(
                        stock=stock,
                        order_percent=order_percent,
                    )
                },
            },
        }
        tasks.append(config)


def run_bt(config):
    file_path = "C:\\Users\wilsonZhang\PycharmProjects\quant_ml\\rqalpha_backtest\Threat_up_tune_parameter\Threat_up.py"
    run_file(file_path, config)

#
for task in tasks:
    run_bt(task)


