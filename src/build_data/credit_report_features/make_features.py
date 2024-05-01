import pandas as pd
import numpy as np


def build_aggregate_credit_report_information_by(df: pd.DataFrame, aggregate_by: str) -> pd.DataFrame:
    """
    Aggregates credit report information by customer ID and a specified category, computing various
    statistical measures for each group. This function creates a wide-format DataFrame where each row
    represents a unique customer and columns represent aggregated metrics of credit-related activities
    across different categories specified by 'aggregate_by'.

    Parameters:
    - df: The input DataFrame containing credit report data.
    - aggregate_by: The column name to further group the data (e.g., 'account_type').

    Returns:
    - pd.DataFrame: A pivot table where the index is 'customer_id', columns are created by the values of
      'aggregate_by', and cells contain aggregated credit report metrics such as sums, medians, and
      standard deviations of financial metrics. Each feature is prefixed with 'credit_reports__' to
      denote its origin from credit report data.

    Examples of aggregated metrics include:
    - Count of inquiries
    - Sum, median, and standard deviation of maximum credit
    - Number of unique credit types
    - Maximum, median, and standard deviation of delayed payment severity
    """

    df_aggregates = df.groupby(["customer_id", aggregate_by]).agg({
        "cdc_inquiry_id": ["count"],
        "max_credit": ['sum', 'median', 'std'],
        "credit_limit": ['sum', 'median', 'std'],
        "current_balance": ['sum', 'median', 'std'],
        "balance_due_worst_delay": ['max', 'median', 'std'],
        "balance_due": ['sum', 'median', 'std'],
        "debt_ratio": ['max', 'median', 'std'],
        "credit_type": ["nunique"],
        "business_type": ["nunique"],
        "age": ['max', 'median', 'std'],
        "severity_delayed_payments": ['max', 'median', 'std'],
        "balance_due_ratio": ['max', 'median', 'std'],
        "balance_due_worst_delay_ratio": ['max', 'median', 'std'],
        "has_delayed_payments": ['sum'],
        "is_individual_responsibility": ['sum'],
        "payment_amount": ['sum']
    })
    df_aggregates.columns = ["_".join(i) for i in df_aggregates.columns.values]
    df_aggregates = df_aggregates.reset_index()

    values = df_aggregates.columns.to_list()
    values.remove("customer_id")
    values.remove(aggregate_by)

    df_pivot = df_aggregates.pivot_table(
        index='customer_id',
        columns=aggregate_by,
        values=values,
        aggfunc='first'
    )

    features = ["credit_reports__" + "_".join(col).lower() for col in df_pivot.columns.values]
    df_pivot.columns = features

    return df_pivot.reset_index()


def build_aggregate_credit_report_information(df: pd.DataFrame,
                                              aggregate_column_names: str = "credit_reports__") -> pd.DataFrame:
    """
    Aggregates key financial indicators from a credit report dataset at the customer level. This function
    computes various statistical metrics such as count, sum, max, min, mean, median, and standard deviation
    for different financial variables to comprehensively summarize each customer's credit activities.

    Parameters:
    - df: The DataFrame containing credit report data with multiple entries per customer.
    - aggregate_column_names: A prefix for the column names in the aggregated DataFrame,
      helping to identify the source of the features. Defaults to 'credit_reports__'.

    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a unique customer_id and columns represent
      aggregated metrics for various credit-related features. Column names are prefixed with the value
      provided in `aggregate_column_names`, followed by the specific aggregation type (e.g., 'sum', 'max').

    Aggregates the following metrics for each customer:
    - Count of credit inquiries
    - Sum of maximum credit extended
    - Sum of credit limits across all accounts
    - Sum of current balances across accounts
    - Maximum and sum of balances due
    - Maximum, median, mean, and standard deviation of the debt ratio
    - Number of unique credit types and business types utilized by the customer
    - Maximum and minimum age of accounts
    - Maximum, median, mean, and standard deviation of severity of delayed payments
    - Aggregated metrics related to balance due ratios
    - Sum of instances where payments were delayed
    - Sum of instances denoting individual responsibility for the credit
    - Sum of payment amounts
    """

    df_aggregates = df.groupby(["customer_id"]).agg({
        "cdc_inquiry_id": ["count"],
        "max_credit": ["sum"],
        "credit_limit": ["sum"],
        "current_balance": ["sum"],
        "balance_due_worst_delay": ['max'],
        "balance_due": ['sum'],
        "debt_ratio": ['max', 'median', 'mean', 'std'],
        "credit_type": ["nunique"],
        "business_type": ["nunique"],
        "age": ['max', 'min'],
        "severity_delayed_payments": ['max', 'median', 'mean', 'std'],
        "balance_due_ratio": ['max', 'median', 'mean', 'std'],
        "balance_due_worst_delay_ratio": ['max', 'median', 'mean', 'std'],
        "has_delayed_payments": ['sum'],
        "is_individual_responsibility": ['sum'],
        "payment_amount": ['sum']
    })
    df_aggregates.columns = [aggregate_column_names + "_".join(i) for i in df_aggregates.columns.values]
    df_aggregates = df_aggregates.reset_index()

    return df_aggregates


def build_credit_report_features(df_aux: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and enriches a DataFrame containing credit report data by adding derived features,
    aggregating data, and preparing the dataset for further analysis and modeling.

    This function handles:
    - Standardizing column names and data types.
    - Calculating various financial ratios and flags based on credit data.
    - Aggregating credit data at the customer level to provide a holistic view of their credit status.
    - Merging different aggregations to form a comprehensive feature set per customer.

    Parameters:
    - df_aux: The input DataFrame with raw credit report data.

    Returns:
    - pd.DataFrame: A DataFrame indexed by 'customer_id' with new features derived from credit report data,
      including ratios of credit use, payment behaviors, and aggregate metrics of credit activities.
    """

    df = df_aux.copy()
    df.columns = [i.lower() for i in df.columns]
    df["account_type"] = df["account_type"].str.replace(" ", "_")
    df = df.astype({"delayed_payments": "float"})
    df[["responsability_type", "credit_type", "business_type"]]

    df = df.assign(
        age=np.where(
            df["loan_opening_date"].isnull(), np.nan, np.where(
                df["loan_closing_date"].isnull(), (df["application_datetime"] - df["loan_opening_date"]).dt.days,
                (df["loan_closing_date"] - df["loan_opening_date"]).dt.days)),
        is_opening=np.where(
            df["loan_closing_date"].isnull(), 1, np.where(~df["loan_closing_date"].isnull(), 0, np.nan)),
        debt_ratio=(df["current_balance"] / df["max_credit"]).replace([np.inf, -np.inf], np.nan),
        severity_delayed_payments=(df["delayed_payments"] / df["total_payments"]).replace([np.inf, -np.inf], np.nan),
        balance_due_ratio=(df["balance_due"] / df["max_credit"]).replace([np.inf, -np.inf], np.nan),
        balance_due_worst_delay_ratio=(df["balance_due_worst_delay"] / df["max_credit"]).replace([np.inf, -np.inf],
                                                                                                 np.nan),
        has_delayed_payments=np.where(df["delayed_payments"] > 0, 1, np.where(df["delayed_payments"] == 0, 0, np.nan)),
        is_individual_responsibility=np.where(df["responsability_type"] == "INDIVIDUAL (TITULAR)", 1,
                                              np.where(~df["responsability_type"].isnull(), 0, np.nan))
    )

    agg_df = build_aggregate_credit_report_information(df).rename(columns={
        "credit_reports__cdc_inquiry_id_count": "credit_reports__loans_count",
        "credit_reports__is_opening_sum": "credit_reports__opening_loan_count",
        "credit_reports__has_delayed_payments_sum": "credit_reports__loans_with_at_least_one_delayed_count",
    })

    df_aux = df[df["is_opening"] == 1]
    agg_df_open_loans = build_aggregate_credit_report_information(df_aux,
                                                                  aggregate_column_names="credit_reports__open_loans_").rename(
        columns={
            "credit_reports__open_loans_cdc_inquiry_id_count": "credit_reports__open_loans_count",
            "credit_reports__open_loans_is_opening_sum": "credit_reports__opening_loan_count",
            "credit_reports__open_loans_has_delayed_payments_sum": "credit_reports__open_loans_with_at_least_one_delayed_count",
        })

    agg_df_by_credit_type = build_aggregate_credit_report_information_by(df, aggregate_by="account_type")

    df_pivot = df[["customer_id"]].drop_duplicates()
    df_pivot = pd.merge(df_pivot, agg_df, how="left", on="customer_id")
    df_pivot = pd.merge(df_pivot, agg_df_open_loans, how="left", on="customer_id")
    df_pivot = pd.merge(df_pivot, agg_df_by_credit_type, how="left", on="customer_id")

    df_pivot = df_pivot.assign(
        credit_reports__opening_loans_ratio=df_pivot["credit_reports__open_loans_count"] / df_pivot[
            "credit_reports__loans_count"],
        credit_reports__loans_with_at_least_one_delayed_ratio=df_pivot[
                                                                  "credit_reports__loans_with_at_least_one_delayed_count"] /
                                                              df_pivot["credit_reports__loans_count"],
        credit_reports__debt_ratio=df_pivot["credit_reports__balance_due_sum"] / df_pivot[
            "credit_reports__max_credit_sum"],
        credit_reports__debt_due_ratio=df_pivot["credit_reports__balance_due_sum"] / df_pivot[
            "credit_reports__balance_due_sum"]
    )

    return df_pivot