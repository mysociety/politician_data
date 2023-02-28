from pathlib import Path
import requests
import json
from rich import print
import pandas as pd

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
    person_df = person_df.drop(columns=["identifiers", "other_names", ])

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


def main():
    download_people()
    flatten_data()
