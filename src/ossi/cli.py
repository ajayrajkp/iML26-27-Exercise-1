import argparse
from .experiment import BANK_DATA_PATH, DATASETS, DEFAULT_SEEDS, run

def main():
    parser = argparse.ArgumentParser(description="Run repeated OSSI structural workload experiments")
    parser.add_argument("--output", default="results", help="directory for CSVs, JSON traces, and plots")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="breast_cancer")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="one or more split/model seeds; default: 42 43 44 45 46")
    parser.add_argument("--bank-data", default=str(BANK_DATA_PATH), help="path to UCI bank-full.csv for the bank_marketing dataset")
    args = parser.parse_args()
    per_run, summary = run(args.output, args.dataset, args.seeds, bank_data_path=args.bank_data)
    print(f"Completed {len(per_run)} model runs. Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
