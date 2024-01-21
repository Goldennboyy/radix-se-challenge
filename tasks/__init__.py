"""Package tasks."""

from invoke import Collection

from .logging import configure_root_logger
from .tasks import lint, run, serve, test

configure_root_logger()

ns = Collection()
ns.add_task(lint)
ns.add_task(test)
ns.add_task(serve)
ns.add_task(run)
