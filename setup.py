from setuptools import setup, find_packages  # type: ignore

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="followthemoney-predict",
    version="0.1.1",
    author="Organized Crime and Corruption Reporting Project",
    author_email="data@occrp.org",
    url="https://github.com/alephdata/followthemoney-predict/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(
        exclude=[
            "ez_setup",
            "examples",
            "tests",
            "cache",
            "data",
            "experiments",
            "models",
        ]
    ),
    namespace_packages=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ftm-predict = followthemoney_predict.cli:main",
            "followthemoney-predict = followthemoney_predict.cli:main",
        ],
    },
    tests_require=["coverage", "nose"],
)
