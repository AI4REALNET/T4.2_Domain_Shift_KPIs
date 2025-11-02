import os
import setuptools
from setuptools import setup
from domain_shift_kpis import get_version

def get_data_files(directory):
    my_list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            my_list.append((dirpath, [os.path.join(dirpath,f)]))
    return my_list

pkgs = {
    "required": [
        "numpy",
        "gymnasium",
        "l2rpn_baselines"
    ],
    "extras": {
        "powergrid": [
            "grid2op==1.9.8",
            "pybind11==2.8.1",
            "lightsim2grid==0.8.1",
            "leap-net==0.0.5",
            "protobuf==3.20.2",
            "pandapower==2.11.0",
            "pandas==1.5.3",
            "jupyter",
            "tensorflow==2.8.1",
            "torch==2.0.1",
            "imageio==2.34.0",
            "plotly==5.20.0"
        ],
        "railway": [
            "Flatland"
        ],
        "atm": [
            "BlueSky"  
        ],
    }
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='domain_shift_kpis',
      version=get_version("__init__.py"),
      description='Domain shift KPIs',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='Domain shift, KPIs, Adaptation time, RL',
      author='Milad Leyli-abadi',
      author_email='milad.leyli-abadi@irt-systemx.fr',
      url="https://github.com/AI4REALNET/T4.2_Domain_Shift_KPIs.git",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini"],
            },
    #   data_files=get_data_files("configurations"),#[("configurations/powergrid/benchmarks/", ["configurations/powergrid/benchmarks/l2rpn_case14_sandbox.ini"])],
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)