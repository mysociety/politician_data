import rich_click as click
from .download import download_people, flatten_data


@click.group()
def cli():
    pass


def main():
    cli()


@cli.command()
def download():
    download_people()


@cli.command()
def flatten():
    flatten_data()


@cli.command()
def example():
    print("This is an example function")


if __name__ == "__main__":
    main()
