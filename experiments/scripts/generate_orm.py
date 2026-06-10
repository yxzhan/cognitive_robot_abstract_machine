import logging
from pathlib import Path

import experiments
from experiments.orm import ormatic_interface
import pycram.orm.ormatic_interface

from krrood.ormatic.ormatic import ORMatic

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package(
    [experiments], [pycram.orm.ormatic_interface], [], type_mappings={}
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = Path(ormatic_interface.__file__)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
