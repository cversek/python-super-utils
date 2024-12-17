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
    author="Craig Versek",
    author_email="cversek@gmail.com",
    description="A utility package for debugging and tagged logging.",
    url="https://github.com/cversek/python-super-utils",
    project_urls={
        "Source": "https://github.com/cversek/python-super-utils",
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debugging",
    ],
)
