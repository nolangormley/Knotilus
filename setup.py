from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Knotilus',
    version='0.1.0',
    description='Knotilus: A differentiable piecewise linear regression framework',
    long_description=readme,
    author='Nolan Gormley',
    author_email='nolangormley@gmail.com',
    url='https://github.com/dagormz/Knotilus',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)