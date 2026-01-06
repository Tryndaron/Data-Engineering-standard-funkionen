from setuptools import setup, find_packages



setup(
    name="PandasDFA",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "openpyxl"  # falls du Abhängigkeiten hast
    ],
)






