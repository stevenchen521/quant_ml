import glob
import pandas as pd

results = []

for name in glob.glob("result\*.pkl"):
    result_dict = pd.read_pickle(name)
    summary = result_dict["summary"]
    results.append({
        "name": name,
        "annualized_returns": summary["annualized_returns"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
    })

results_df = pd.DataFrame(results)

print("-" * 50)
print("Sort by sharpe")
print(results_df.sort_values("sharpe", ascending=False)[:10])


print("-" * 50)
print("Sort by annualized_returns")
print(results_df.sort_values("annualized_returns", ascending=False)[:10])


