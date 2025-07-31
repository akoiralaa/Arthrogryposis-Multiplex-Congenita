import pandas as pd

def write_summary_csv(rows, filename):
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
