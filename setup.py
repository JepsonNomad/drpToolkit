from setuptools import setup

setup(
    name='drpToolkit',
    url='https://github.com/jepsonnomad/drpToolkit',
    author='Christian John',
    author_email='cjohn@ucdavis.edu',
    packages=['drpToolkit'],
    package_dir={'drpToolkit': 'drpToolkit'},
    install_requires=['numpy'],
    version='0.0.0.9000',
    license='Gnu GPL 3.0',
    description='Digital repeat photography imagery management and analysis',
    long_description=open('README.md').read(),
    entry_points = {
        'console_scripts': ['drpPrep=drpToolkit.prep:main',
        'drpAlign=drpToolkit.align:main',
        'drpExtract=drpToolkit.extract:main',
        'drpPanelize=drpToolkit.panelize:main']
    }
)