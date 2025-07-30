from setuptools import setup, find_packages

setup(
    name="cpfd_rom",
    version="0.1",
    packages=find_packages(include=["cpfd_rom", "ml_rom", "pca_rbf_rom", "util"]),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "pyyaml",
        "absl-py"
    ],
    entry_points={
        "console_scripts": [
            "rom-cli=cpfd_rom.rom_cli:main"
        ]
    },
    author="Saurav Mitra",
    description="A CLI tool to run reduced-order models from CFD datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown"
)
