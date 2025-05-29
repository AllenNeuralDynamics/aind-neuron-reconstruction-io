import os

from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setup(
        name="aind-neuron-reconstruction-io",
        version=get_version("src/aind_neuron_reconstruction_io/__init__.py"),
        author="Matt Mallory, Sharmishtaa Seshamani",
        author_email="matt.mallory@alleninstitute.org, sharmishtaas@alleninstitute.org",
        packages=find_packages(),
        package_data={
            "aind_neuron_reconstruction_io": ["util_files/*.nrrd"]
        },
        include_package_data=True,
    )
