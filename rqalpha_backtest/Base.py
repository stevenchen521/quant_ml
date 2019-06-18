import pandas as pd



def pickle_to_excel(pickle_path, excel_path = None):
    if excel_path == None:
        excel_path = pickle_path.replace('.pkl', '.xlsx', inplace = True)
    else:
        pass
    writer = pd.ExcelWriter(excel_path)
    p1 = pd.read_pickle(pickle_path)
    p1.benchmark_portfolio.to_excel(writer, sheet_name="benchmark_portfolio")
    p1.plots.to_excel(writer, sheet_name="plots")
    p1.portfolio.to_excel(writer, sheet_name="portfolio")
    p1.stock_account.to_excel(writer, sheet_name="stock_account")
    p1.stock_positions.to_excel(writer, sheet_name="stock_positions")
    # .to_excel(writer, sheet_name="QgrC_top")
    for key, val in p1.summary():
        writer.writerow([key, val])
    p1.trades.to_excel(writer, sheet_name="trades")
    writer.save()