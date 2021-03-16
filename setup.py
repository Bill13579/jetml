import setuptools

with open("README_PYPI.md", "r") as rm:
    long_description = rm.read()

setuptools.setup(
    name="jetml",
    version="0.1",
    author="BillK",
    author_email="bluesky42624@gmail.com",
    description="A powerful & extensive machine learning library focused on readability and completeness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bill13579/jetml",
    packages=setuptools.find_packages(),
    install_requires=("jetmath",),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)

