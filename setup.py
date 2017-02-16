from setuptools import setup

setup(name="skymapper",
      description="Mapping astronomical survey data on the sky, handsomely",
      long_description="Mapping astronomical survey data on the sky, handsomely",
      version="0.1",
      license="MIT",
      author="Peter Melchior",
      author_email="peter.m.melchior@gmail.com",
      url="https://github.com/pmelchior/skymapper",
      requires=["matplotlib", "numpy"],
      packages=["skymapper", "skymapper.surveys"],
      package_data={"skymapper.surveys": ['*.txt']}
)
