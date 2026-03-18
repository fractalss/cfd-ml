#!/usr/bin/env python3
import os
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize


PACKAGE_NAME = "cpfd_rom"
VERSION = "0.3"

# Keep no readable source modules in the final wheel except __init__.py files.
KEEP_PY = set()


def list_py_files(base_dir: str):
    py_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            if f in KEEP_PY:
                continue
            py_files.append(os.path.join(root, f))
    return py_files


class build_py_strip_sources(_build_py):
    """
    Build python package into build/lib, then remove non-__init__.py sources so the wheel
    ships only compiled extensions (.so) + __init__.py package markers.
    """

    def run(self):
        super().run()

        pkg_root = os.path.join(self.build_lib, PACKAGE_NAME)
        if os.path.isdir(pkg_root):
            for root, _, files in os.walk(pkg_root):
                for f in files:
                    if f.endswith(".py") and f != "__init__.py" and f not in KEEP_PY:
                        os.remove(os.path.join(root, f))


py_sources = list_py_files(PACKAGE_NAME)
extensions = []

for src in py_sources:
    modname = Path(src).with_suffix("").as_posix().replace("/", ".")
    extensions.append(Extension(modname, [src]))

ext_modules = cythonize(
    extensions,
    compiler_directives={"language_level": "3"},
    build_dir="build/cython",
)


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Saurav Mitra",
    description="AI-driven Reduced Order Modeling (ROM) tool for Barracuda CFD data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=[f"{PACKAGE_NAME}", f"{PACKAGE_NAME}.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_py": build_py_strip_sources},
    include_package_data=False,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "rom-cli-bin=cpfd_rom.rom_cli:main",
        ]
    },
    install_requires=[
        "torch==2.2.2",
        "torch-geometric==2.7.0",
        "scikit-learn==1.7.2",
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scipy==1.16.3",
        "matplotlib==3.10.7",
        "tqdm==4.67.1",
        "networkx==3.5",
        "pyyaml==6.0.3",
    ],
    extras_require={
        "lagrangian": [
            "torch_scatter",
            "torch_sparse",
            "torch_cluster",
            "torch_spline_conv",
        ]
    },
    python_requires=">=3.10,<3.13",
    exclude_package_data={"": ["*.c", "*.cpp", "*.h", "*.pyx", "*.pxd"]},
)