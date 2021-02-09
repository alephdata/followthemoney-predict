from setuptools import setup, find_packages  # type: ignore

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="xref_experiments",
    version="0.0.1",
    author="OCCRP",
    author_email="micha@occrp.org",
    url="https://github.com/alephdata/xref-experiments",
    license="MIT",
    packages=find_packages(exclude=["ez_setup", "examples", "test", "data"]),
    namespace_packages=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    tests_require=["coverage", "nose"],
)
