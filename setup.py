from setuptools import setup, find_packages
import os
import io


HERE = os.path.dirname(os.path.abspath(__file__))


def read(*parts, **kwargs):
    filepath = os.path.join(HERE, *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_requirements(path):
    content = read(path)
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


# setup_requires = ["numpy"]

install_requires = get_requirements("requirements.txt")

setup(
    name="DIGDriver",
    version="0.2.0",
    description="Flexible cancer driver element detection",
    author="Maxwell Sherman",
    author_email="msherman997@gmail.com",
    url="",
    packages=find_packages(),
    # packages=["DIGDriver"],
    # setup_requires=setup_requires,
    install_requires=install_requires,
    scripts=["scripts/DataExtractor.py",
             "scripts/DigPretrain.py",
             "scripts/DigPreprocess.py",
             "scripts/mutationFunction.R",
             "scripts/DigDriver.py"
            ],
    include_package_data=True,
    package_data={'': ['data/*.txt']},
    # entry_points={"console_scripts": ["clodius = clodius.cli.aggregate:cli"]},
)
