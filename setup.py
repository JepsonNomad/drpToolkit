from setuptools import setup

setup(
    name='drpToolkit',
    url='https://github.com/jepsonnomad/drpToolkit',
    author='Christian John',
    author_email='cjohn@ucdavis.edu',
    packages=['drpToolkit'],
    install_requires=['numpy'],
    version='0.0.1',
    license='Gnu GPL 3.0',
    description='Digital repeat photography imagery management and analysis',
    long_description=open('README.md').read(),
    scripts=['drpToolkit/prep','drpToolkit/align','drpToolkit/extract','drpToolkit/panelize']
)