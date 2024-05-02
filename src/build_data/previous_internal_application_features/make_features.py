import pandas as pd


def build_previous_internal_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a Main Dataset to create features derived from the customer's history within the organization.
    These features include:
        - Ratio of approved BNPL applications to total BNPL applications.
        - Days between the last BNPL application and the credit application date.
        - Days between the first BNPL application and the credit application date.
        - Days from account creation to credit application.
        - Total counts of SF and BNPL applications, including approved BNPL applications.
        - Number of inquiries to credit reports from external entities in the last 3 and 6 months.


    Parameters:
    df (pd.DataFrame): A DataFrame containing the main dataset with the customer's history within the organization.

    Returns:
    pd.DataFrame: A DataFrame containing the loan ID and the newly created features prefixed with 'previous_internal_apps__'.
    """

    df.columns = [i.lower() for i in df.columns]

    df = df.assign(
        previous_internal_apps__ratio_bnpl_approved=(df["n_bnpl_approved_apps"] / df["n_bnpl_apps"]).fillna(0),
        previous_internal_apps__last_bnpl_app_to_application_days=(
                    df["application_datetime"] - df["first_bnpl_app_date"]).dt.days,
        previous_internal_apps__first_bnpl_app_to_application_days=(
                    df["application_datetime"] - df["last_bnpl_app_date"]).dt.days,
        previous_internal_apps__account_to_application_days=df["account_to_application_days"],
        previous_internal_apps__n_sf_apps=df["n_sf_apps"].fillna(0),
        previous_internal_apps__n_bnpl_apps=df["n_bnpl_apps"].fillna(0),
        previous_internal_apps__n_bnpl_approved_apps=df["n_bnpl_approved_apps"].fillna(0),
        previous_internal_apps__n_inquiries_l3m=df["n_inquiries_l3m"].fillna(0),
        previous_internal_apps__n_inquiries_l6m=df["n_inquiries_l6m"].fillna(0),
    )

    features = [i for i in df.columns if "previous_internal_apps__" in i]

    return df[["loan_id"] + features]