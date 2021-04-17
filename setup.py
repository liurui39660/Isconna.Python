from os.path import exists

from setuptools import setup

if exists("VERSION"):  # GitHub Actions
	with open("VERSION") as file:
		version = file.readline()
elif exists("PKG-INFO"):  # Conda
	with open("PKG-INFO") as file:
		for line in file:
			if line.startswith("Version: "):
				version = line[9:]
				break
else: raise FileNotFoundError("Can't find the version number")

setup(
	name="Isconna",
	version=version,
	author="Rui LIU",
	author_email="xxliuruiabc@gmail.com",
	description="Python porting of the Isconna algorithm",
	long_description="Pypi does not support an AsciiDoc README<br>Please visit our GitHub repo https://github.com/liurui39660/Isconna.Python",
	long_description_content_type="text/markdown",
	url="https://github.com/liurui39660/Isconna.Python",
	project_urls={},
	classifiers=[
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: Apache Software License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3"
	],
	package_dir={"": "src"},
	packages=["Isconna"],
	install_requires=["numba", "numpy"],
	python_requires=">=3.6",
)
