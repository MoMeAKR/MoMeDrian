from setuptools import setup, find_packages

setup(
    name="mome_drian",
    version="0.1.0",
    description="The lib that shows i'm an artist.",
    author="mOmE",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": [
            "js_utils/*.js",
            "mOmEdRiAn_available_tools.json",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
