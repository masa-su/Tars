import io,os,re
from setuptools import setup, find_packages

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
    description='Variational autoencoder library',
    url='https://github.com/masa-su/Tars',
    author='Masahiro Suzuki',
    install_requires=[
        'Lasagne',
        'matplotlib',
        'progressbar2',
        'Theano',
        'sklearn'
    ]
)
