try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='moonshot',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Automatic crater detection for Moon and Mars.',
      author='ACDS project',
      packages=['moonshot']
      )
