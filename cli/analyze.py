import pandas as pd
from datasci_automation.core.pipeline import run_pipeline
import sys

def main():
    df = pd.read_csv(sys.argv[1])
    result = run_pipeline(df)
    print(result)

if __name__ == "__main__":
    main()
