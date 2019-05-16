from icemet_nn import __version__

from setuptools import setup

with open("README.md") as fp:
	readme = fp.read()

with open("requirements.txt") as fp:
	requirements = fp.read().splitlines()

setup(
	name="icemet-nn",
	version=__version__,
	py_modules=["icemet_nn"],
	
	install_requires=requirements,
	
	author="Eero Molkoselkä",
	author_email="eero.molkoselka@gmail.com",
	description="ICEMET neural network tools",
	long_description=readme,
	url="https://github.com/molkoback/icemet-nn",
	license="MIT",
	
	entry_points={
		"console_scripts": [
			"icemet-nn-train = icemet_nn:train_main",
			"icemet-nn-test = icemet_nn:test_main"
		]
	},
	
	classifiers=[
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Atmospheric Science"
	]
)
