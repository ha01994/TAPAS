from pathlib import Path
import pandas as pd


def peptide_from_pmhc(s: str) -> str:
    return str(s).split("_", 1)[0]


def load_splits(base: Path, fold: int):
    paths = {
        "train": base / f"fold{fold}_train.csv",
        "val": base / f"fold{fold}_val.csv",
        "test": base / f"fold{fold}_test.csv",
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    return {k: pd.read_csv(p) for k, p in paths.items()}


def main() -> None:
    root = Path(__file__).resolve().parent

    for tag, base, show_pep in (
        ("rs", root / "dataset_iptm_filtered_rs", False),
        ("ss", root / "dataset_iptm_filtered_ss", True),
    ):
        print(f"\n[{tag}] {base.name}/")
        if show_pep:
            print("fold,train,val,test,train_pep,val_pep,test_pep")
        else:
            print("fold,train,val,test")

        for fold in range(5):
            dfs = load_splits(base, fold)
            if dfs is None:
                print(f"{fold},(missing csv)")
                continue
            n_tr, n_va, n_te = len(dfs["train"]), len(dfs["val"]), len(dfs["test"])
            row = f"{fold},{n_tr},{n_va},{n_te},"
            if show_pep:
                pe = [
                    dfs[s]["pmhc"].map(peptide_from_pmhc).nunique()
                    for s in ("train", "val", "test")
                ]
                row += f"{pe[0]},{pe[1]},{pe[2]}"
            print(row)
    print()

    for tag, base, show_pep in (
        ("rs", root / "dataset_rs", False),
        ("ss", root / "dataset_ss", True),
    ):
        print(f"\n[{tag}] {base.name}/")
        if show_pep:
            print("fold,train,val,test,train_pep,val_pep,test_pep")
        else:
            print("fold,train,val,test")

        for fold in range(5):
            dfs = load_splits(base, fold)
            if dfs is None:
                print(f"{fold},(missing csv)")
                continue
            n_tr, n_va, n_te = len(dfs["train"]), len(dfs["val"]), len(dfs["test"])
            row = f"{fold},{n_tr},{n_va},{n_te},"
            if show_pep:
                pe = [
                    dfs[s]["pmhc"].map(peptide_from_pmhc).nunique()
                    for s in ("train", "val", "test")
                ]
                row += f"{pe[0]},{pe[1]},{pe[2]}"
            print(row)
    print()

    
    
    
if __name__ == "__main__":
    main()
