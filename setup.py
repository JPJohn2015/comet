from setuptools import setup, find_packages

setup(
    name="comet",
    version="0.1.0",
    author='James Johnson',
    author_email='jp_johnson@comcast.net',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "attrs",
        "poliastro",
        "pytest"
    ],
)