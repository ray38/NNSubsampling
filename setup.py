import setuptools

with open("README.md", "r") as fh:

	long_description = fh.read()

setuptools.setup(

	name='NNSubsampling',  

	version='0.1',

	packages=['NNSubsampling'],

	author="Xiangyun (Ray) Lei",

	author_email="xlei38@gatech.edu",

	description="This is a sampling algorithm aiming to achieve near-uniform sampling of high dimentional data sets",

	long_description=long_description,

	long_description_content_type="text/markdown",

	url="https://github.com/ray38/NNSubsampling",

	classifiers=[
	
		"Programming Language :: Python :: 2",

		"Programming Language :: Python :: 3",

		"License :: OSI Approved :: MIT License",

		"Operating System :: OS Independent",

	],
	
	install_requires=[
		'numpy',
		'scipy',
		'scikit-learn'
	]

 )
