from pathlib import Path
import requests
import json
from rich import print
import pandas as pd
from typing import TypeVar

PEOPLE_SOURCE = (
    "https://raw.githubusercontent.com/mysociety/parlparse/master/members/people.json"
)


def wrap_if_dict(v: list | dict):
    if isinstance(v, dict):
        return [v]
    return v


def identifer_expansion(
    df: pd.DataFrame, column_name: str, id_col_name: str
) -> pd.DataFrame:
    """Expand a column of lists of dicts into a dataframe of dicts"""
    identfiers = df[lambda x: ~x[column_name].isna()].apply(
        lambda row: [
            dict(identifier, **{id_col_name: row["id"]})
            for identifier in wrap_if_dict(row[column_name])
        ],
        axis=1,
    )
    # flatten the list of list of dicts into single list
    identifers = [item for sublist in identfiers for item in sublist]
    new_df = pd.DataFrame(identifers)
    # move id column to front
    new_df = new_df[
        [id_col_name] + [col for col in new_df.columns if col != id_col_name]
    ]
    # sort by ID
    new_df = new_df.sort_values(id_col_name)
    return new_df


def download_people():
    """
    Download people json to data/raw
    """
    print("[blue]Downloading people.json[/blue]")
    people = requests.get(PEOPLE_SOURCE).json()
    Path("data", "raw", "people.json").write_text(json.dumps(people, indent=2))


def get_names() -> pd.DataFrame:

    df = pd.read_parquet(
        Path(
            "data", "packages", "uk_politician_data", "person_alternative_names.parquet"
        )
    )
    df = df[df["note"] == "Main"]
    df = df.drop(columns=["note"])
    df["last_name"] = df["family_name"].fillna(df["lordname"])
    df["nice_name"] = df.apply(
        lambda x: (
            f"{x['given_name']} {x['last_name']}"
            if pd.isna(x["honorific_prefix"])
            else f"{x['given_name']} {x['last_name']} ({x['honorific_prefix']})"
        ),
        axis=1,
    )

    df = df[["person_id", "given_name", "last_name", "nice_name"]]
    return df


def create_reduced_membership_table():
    """
    Create a membership table that brings in most of the details used in the pw_mp table
    """
    package_dir = Path("data", "packages", "uk_politician_data")

    member_df = pd.read_parquet(Path(package_dir, "memberships.parquet"))

    post_df = pd.read_parquet(Path(package_dir, "posts.parquet"))

    names_df = get_names()

    df = member_df.merge(
        names_df, left_on="person_id", right_on="person_id", how="left"
    )

    df = df.merge(post_df, left_on="post_id", right_on="id", how="left")

    # organization_id_x and organization_id_y should be merged
    # they are mutually exclusive and one will be none for each row

    short_chamber = {
        "house-of-commons": "commons",
        "house-of-lords": "lords",
        "northern-ireland-assembly": "ni",
        "scottish-parliament": "scotland",
        "welsh-parliament": "wales",
        "crown": "crown",
    }

    df["chamber"] = (
        df["organization_id_x"].fillna(df["organization_id_y"]).map(short_chamber)
    )
    df["label"] = df["label_x"].fillna(df["label_y"])
    df["role"] = df["role_x"].fillna(df["role_y"])
    df
    allowed_cols = {
        "id_x": "membership_id",
        "person_id": "person_id",
        "area_name": "constituency",
        "start_date_x": "start_date",
        "end_date_x": "end_date",
        "start_reason": "start_reason",
        "end_reason": "end_reason",
        "on_behalf_of_id": "party",
        "chamber": "chamber",
        "label": "label",
        "role": "role",
        "given_name": "first_name",
        "last_name": "last_name",
        "nice_name": "nice_name",
    }

    df = df.rename(columns=allowed_cols)
    df = df[allowed_cols.values()]  # type: ignore

    df.to_parquet(Path(package_dir, "simple_memberships.parquet"), index=False)


str_or_none = TypeVar("str_or_none", str, None)


def fix_partial_date(date: str_or_none) -> str_or_none:
    """
    Some dates are in the format YYYY, or YYYY-MM. This function converts
    them to YYYY-MM-DD.
    """
    if date is None:
        return date
    if len(date) == 4:
        return date + "-01-01"
    elif len(date) == 7:
        return date + "-01"
    else:
        return date


def minus_one_date(date: str) -> str:
    """
    Give one day before an ISO date
    """
    # special case for 9999 - don't need to do day before
    if date == "9999-12-31":
        return date
    return (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def create_membership_counts():
    """
    Create a dataset of membership counts and time ranges
    """
    package_dir = Path("data", "packages", "uk_politician_data")

    df = pd.read_parquet(package_dir / "simple_memberships.parquet")

    df["start_date"] = df["start_date"].apply(fix_partial_date)
    df["end_date"] = df["end_date"].apply(fix_partial_date)

    # We want to create a new dataframe with value, date, chamber
    # we are converting the start and end dates into a list of events
    # if there is a start date, there is a value of 1
    # if there is an end date, there is a value of -1

    # we are going to use the melt function to do this
    ndf = pd.melt(
        df,
        id_vars=["chamber"],
        value_vars=["start_date", "end_date"],
        var_name="event",
        value_name="date",
    ).sort_values(["chamber", "date"])

    ndf["event"] = ndf["event"].replace({"start_date": 1, "end_date": -1})

    allowed_chambers = ["commons", "lords", "scotland", "wales", "ni"]

    def get_range_counts(df: pd.DataFrame) -> pd.DataFrame:
        # remove none values - our end events that aren't interesting
        df = df[df["date"].notna()].copy(deep=True)
        # use cum sum to get the number of members at any given time
        # obvs this is wrong in the early days
        df["members_count"] = df["event"].cumsum()
        # reduce to unique dates - get the last members count for each date
        df = df.drop_duplicates("date", keep="last")
        # reexpress this as ranges e.g. start_date, end_date, members_count
        df["end_date"] = (
            df["date"].shift(-1, fill_value="9999-12-31").apply(minus_one_date)
        )
        df = df[["date", "end_date", "members_count"]]
        df = df.rename(columns={"date": "start_date"})
        return df

    dfs = []

    for chamber, chamber_df in ndf.groupby("chamber"):
        if chamber in allowed_chambers:
            range_df = get_range_counts(chamber_df)
            range_df["chamber"] = chamber
            dfs.append(range_df)

    final = pd.concat(dfs)
    final.to_parquet(package_dir / "membership_counts.parquet", index=False)


def flatten_data():
    """
    Extract flat tables from people.json
    """
    pol_path = Path("data", "raw", "people.json")
    package_path = Path("data", "packages", "uk_politician_data")

    pol_data: dict = json.loads(pol_path.read_text())
    membership_df = pd.DataFrame(pol_data["memberships"]).sort_values("id")
    identifier_df = identifer_expansion(membership_df, "identifiers", "membership_id")
    membership_df = membership_df.drop(
        columns=["identifiers", "name"]
    )  # there's a (redundant?) name column with only a few entries
    # move 'id', 'person_id', 'organization_id', 'on_behalf_of_id' to front
    priority = ["id", "person_id", "organization_id", "on_behalf_of_id"]
    membership_df = membership_df[
        priority + [col for col in membership_df.columns if col not in priority]
    ]
    orgs_df = pd.DataFrame(pol_data["organizations"]).sort_values("id")
    org_identifiers_df = identifer_expansion(orgs_df, "identifiers", "organization_id")
    orgs_df = orgs_df.drop(columns=["identifiers"])

    person_df = pd.DataFrame(pol_data["persons"]).sort_values("id")
    shortcuts_df = identifer_expansion(person_df, "shortcuts", "person_id")
    person_df = person_df.merge(
        shortcuts_df, left_on="id", right_on="person_id", how="left"
    ).drop(columns=["shortcuts", "person_id"])

    person_identifiers_df = identifer_expansion(person_df, "identifiers", "person_id")
    person_identifiers_df["identifier"] = person_identifiers_df["identifier"].astype(
        str
    )
    alt_names_df = identifer_expansion(person_df, "other_names", "person_id")
    person_df = person_df.drop(
        columns=[
            "identifiers",
            "other_names",
        ]
    )

    posts = pd.DataFrame(pol_data["posts"]).sort_values("id")
    post_identifiers = identifer_expansion(posts, "identifiers", "post_id")
    areas = identifer_expansion(posts, "area", "post_id").rename(
        columns={"name": "area_name"}
    )
    posts = posts.merge(areas, left_on="id", right_on="post_id", how="left").drop(
        columns=["post_id", "identifiers", "area"]
    )

    # write out to csv
    print("[blue]Writing membership to file[/blue]")
    membership_df.to_parquet(package_path / "memberships.parquet", index=False)
    print("[blue]Writing membership identifiers to file[/blue]")
    identifier_df.to_parquet(
        package_path / "membership_identifiers.parquet", index=False
    )
    print("[blue]Writing organizations to file[/blue]")
    orgs_df.to_parquet(package_path / "organizations.parquet", index=False)
    print("[blue]Writing organization identifiers to file[/blue]")
    org_identifiers_df.to_parquet(
        package_path / "organization_identifiers.parquet", index=False
    )
    print("[blue]Writing persons to file[/blue]")
    person_df.to_parquet(package_path / "persons.parquet", index=False)
    print("[blue]Writing person identifiers to file[/blue]")
    person_identifiers_df.to_parquet(
        package_path / "person_identifiers.parquet", index=False
    )
    print("[blue]Writing person alternative names to file[/blue]")
    alt_names_df.to_parquet(
        package_path / "person_alternative_names.parquet", index=False
    )
    print("[blue]Writing posts to file[/blue]")
    posts.to_parquet(package_path / "posts.parquet", index=False)
    print("[blue]Writing post identifiers to file[/blue]")
    post_identifiers.to_parquet(package_path / "post_identifiers.parquet", index=False)
    print("[blue]Writing reduced membership table[/blue]")
    create_reduced_membership_table()
    print("[blue]Writing membership counts[/blue]")
    create_membership_counts()


def main():
    download_people()
    flatten_data()
