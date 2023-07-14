from setuptools import setup, find_packages

setup(
    name="senator_matcher",
    description="A package for matching user input to senators based on their initiatives profiles.",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "stanza",
    ],
)
