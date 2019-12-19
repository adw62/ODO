from setuptools import setup

setup(name='ODO',
      version='0.1',
      description='machine learning drug optimizer',
      url='',
      author='AW',
      author_email='None',
      license='None',
      packages=['ODO'],
      entry_points = {'console_scripts':['ODO = ODO.cli:main']})
