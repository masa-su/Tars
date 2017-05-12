import io
import os
import re
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='Tars',
    version=find_version("Tars", "__init__.py"),
    description='Deep generative model library',
    url='https://github.com/masa-su/Tars',
    author='Masahiro Suzuki',
    dependency_links=[
        'git+https://github.com/Lasagne/Lasagne.git#egg=lasagne-0.2.dev1',
    ],
    install_requires=[
        'Lasagne==0.2.dev1',
        'matplotlib',
        'progressbar2',
        'Theano==0.8.2',
        'sklearn',
        'six',
        'nose_parameterized',
        'mock',                 # for python 2.7
    ]
)
