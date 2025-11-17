from setuptools import setup, find_packages

setup(
    name="cpfd_rom",
    version="0.2",
    packages=find_packages(include=["cpfd_rom", "cpfd_rom.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "pyyaml",
        "torch_geometric",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "rom-cli-bin=cpfd_rom.rom_cli:main"
        ]
    },
    author="Saurav Mitra",
    description="A CLI tool to run reduced-order models from CFD datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown"
)
