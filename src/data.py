"""
Data
"""
import os
import pandas as pd


def get_train(file: str = "bank-train.csv", path: str = "input") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file)).rename(columns={"emp.var.rate": "emp_var_rate",
                                                               "cons.price.idx": "cons_price_idx",
                                                               "cons.conf.idx": "cons_conf_idx",
                                                               "nr.employed": "nr_employed"})
    df = df.astype(dict(
        id="int32",
        age="int8",
        duration="int16",
        campaign="int8",
        pdays="int16",
        previous="int8",
        emp_var_rate="float16",
        cons_price_idx="float32",
        cons_conf_idx="float32",
        euribor3m="float32",
        nr_employed="float32",
        y="bool",
    )).set_index("id")
    # print(df.info(memory_usage="copy"))
    return df


def get_test(file: str = "bank-test.csv", path: str = "input") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file)).rename(columns={"emp.var.rate": "emp_var_rate",
                                                               "cons.price.idx": "cons_price_idx",
                                                               "cons.conf.idx": "cons_conf_idx",
                                                               "nr.employed": "nr_employed"})
    df = df.astype(dict(
        id="int32",
        age="int8",
        duration="int16",
        campaign="int8",
        pdays="int16",
        previous="int8",
        emp_var_rate="float16",
        cons_price_idx="float32",
        cons_conf_idx="float32",
        euribor3m="float32",
        nr_employed="float32",
    )).set_index("id")
    # print(df.info(memory_usage="copy"))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["month"].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                   'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}).astype("int8")
    df["day_of_week"] = df["day_of_week"].map({'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}).astype("int8")
    # print(df.info(memory_usage="copy"))
    return df


def main():
    df = get_train(path="../input")
    df = preprocess(df)
    df.to_parquet("../data/train.parquet", engine="fastparquet")

    print("\ndata types")
    print(df.dtypes)

    print("\nmissing data")
    print(df.isnull().sum())

    print("\nunique values")
    unique = {col: df[col].unique() for col in df.columns if df[col].dtype == "object"}
    for col in unique.keys():
        print(col, unique[col])

    print("\n numeric attributes")
    print(df.describe().transpose())

    df = get_test(path="../input")
    df = preprocess(df)
    df.to_parquet("../data/test.parquet", engine="fastparquet")


if __name__ == "__main__":
    main()
