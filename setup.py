from setuptools import setup

with open("README.md") as file:  # Created in CI
	description = file.read()

with open("VERSION") as file:  # Created in CI
	version = file.readline()

setup(
	name="Isconna",
	version=version,
	author="Rui LIU",
	author_email="xxliuruiabc@gmail.com",
	description="Python porting of the Isconna algorithm",
	long_description=description,
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
