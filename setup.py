from setuptools import setup

setup(
    name='drpToolkit',
    url='https://github.com/jepsonnomad/drpToolkit',
    author='Christian John',
    author_email='cjohn@ucdavis.edu',
    packages=['drpToolkit'],
    install_requires=['numpy'],
    version='1.0.0',
    license='Gnu GPL 3.0',
    description='Digital repeat photography imagery management and analysis',
    long_description=open('README.md').read(),
    scripts=['drpToolkit_scripts/prep','drpToolkit_scripts/align','drpToolkit_scripts/extract','drpToolkit_scripts/panelize']
)