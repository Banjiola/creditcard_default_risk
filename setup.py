from setuptools import setup, find_packages

setup(
    name="creditcard-default-risk",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ipykernel",
        "ucimlrepo>=0.0.7",
        "matplotlib>=3.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "seaborn>=0.13.0",
        "scipy>=1.11.0",
        "joblib>=1.4.0",
        "imbalanced-learn>=0.12.0",
        "scikit-learn>=1.5.0",
        "xgboost>=3.0.4",
    ],
    author="Olabanji Olaniyan",
    description="Credit card default risk prediction project",
    python_requires=">=3.8",
)