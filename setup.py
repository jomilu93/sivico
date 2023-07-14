from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='sivico',
      version="1",
      description="Sivico model, v1",
      license="MIT",
      author="Team Sivico",
      author_email="jmlunamugica@mac.com",
      url="https://github.com/jomilu93/sivico",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
