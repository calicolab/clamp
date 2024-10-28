import click

# TODO: could make it faster but unclear with
# <https://click.palletsprojects.com/en/8.1.x/complex/#lazily-loading-subcommands>
import clamp.clustering
import clamp.embeds
import clamp.pipeline
import clamp.predictions


@click.group(help="Clusterize any model")
def cli():
    pass

cli.add_command(clamp.clustering.main, name="cluster")
cli.add_command(clamp.embeds.main, name="embed")
cli.add_command(clamp.pipeline.main, name="pipeline")
cli.add_command(clamp.predictions.main, name="predict")

if __name__ == "__main__":
    cli()
