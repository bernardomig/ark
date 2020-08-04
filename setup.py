from setuptools import setup, find_namespace_packages

setup(
    name="ark",
    version="0.1.0",
    description="The home of many DL models and training techniques",
    license="Apache 2.0",
    author="Bernardo Lourenço",
    author_email="bernardo.lourenco@ua.pt",
    python_requires=">=3.7.0",
    url="https://github.com/bernardomig/ark",
    packages=find_namespace_packages(
        exclude=["tests", ".tests", "tests_*", "scripts"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
