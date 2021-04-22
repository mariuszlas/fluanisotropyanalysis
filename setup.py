import setuptools

with open("README.md", "r") as rd:
    long_description = rd.read()

setuptools.setup(
        name='fluanisotropyanalysis',
        version='1.0.1',
        author='Mariusz Las, Stuart Warriner',
        author_email='mariusz.las@outlook.com, s.l.warriner@leeds.ac.uk',
        url = 'https://github.com/mariuszlas/Fluorescence-Anisotropy-Analysis',
        description='Read and analyse data from fluorescence anisotropy assays.',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8',
        setup_requires=['wheel'],
        install_requires=['platemapping>=2.0.0','pandas>=1.1.3','numpy>=1.19']
)
