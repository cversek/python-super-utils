from setuptools import setup, find_packages

setup(
    name="python-super-utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "IPython>=7.0",
        "nest_asyncio>=1.5",
        "matplotlib>=3.0",
    ],
)
