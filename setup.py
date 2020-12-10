from setuptools import setup

setup(name='flu_ani_analysis',
        version='0.1.0',
        author='Mariusz Las',
        author_email='cm18mel@leeds.ac.uk',
        url = 'https://github.com/mariuszlas',
        description='Fluorescence anisotropy binding data analysis.',
        packages=['flu_ani_analysis'],
        install_requires=['matplotlib', 'numpy', 'pandas'])