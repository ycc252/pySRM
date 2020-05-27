import setuptools
from setuptools import setup, find_packages

with open("README.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pySRM", # Replace with your own username
    version="0.1.4",
    author="Yijing Yang",
    author_email="yijing.Yang@rub.de",
    license='MIT',
    description="a Python package for the segmented regression model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/ycc252/pySRM",
    packages=setuptools.find_packages("src"),
    package_dir = {"":"src"},
    package_data = {
        "" : ["*.txt","*.info","*.properties"],
        "" : ["data/*.*"],
      },
      exclude = ["*.txt","*.txt","*.txt","*.txt"],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 