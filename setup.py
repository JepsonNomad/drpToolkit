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
    scripts=['drpToolkit_scripts/prep.py','drpToolkit_scripts/align.py','drpToolkit_scripts/extract.py','drpToolkit_scripts/panelize.py']
)