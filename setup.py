from setuptools import setup

setup(name="skymapper",
      description="Mapping astronomical survey data on the sky, handsomely",
      long_description="Mapping astronomical survey data on the sky, handsomely",
      version="0.1",
      license="MIT",
      author="Peter Melchior",
      author_email="peter.m.melchior@gmail.com",
      py_modules=["skymapper"],
      url="https://github.com/pmelchior/skymapper",
      requires=["matplotlib", "numpy"]
)

