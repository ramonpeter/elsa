from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/ramonpeter/elsa"

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="elsa",
    version="1.0.1",
    author="Ramon Winterhalder",
    author_email="ramon.winterhalder@uclouvain.be",
    description="Enhanced Latent Spaces",
    long_description=long_description,
    long_description_content_type="text/rst",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages()
)
