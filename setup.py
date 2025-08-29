from setuptools import setup, find_packages

setup(
    name="model_metrics",
    version="0.0.4a1",
    author="Leonid Shpaner",
    author_email="lshpaner@ucla.edu",
    description="A Python library for model evaluation, performance tracking, and metric visualizations, supporting classification and regression models with robust analytics and reporting.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # Type of the long description
    package_dir={"": "src"},  # Directory where your package files are located
    # Automatically find packages in the specified directory
    packages=find_packages(where="src"),
    project_urls={  # Optional
        "Leonid Shpaner's Website": "https://www.leonshpaner.com",
        "Source Code": "https://github.com/lshpaner/model_metrics/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Classifiers for the package
    python_requires=">=3.7.4",  # Minimum version of Python required
    install_requires=[
        "matplotlib>=3.5.3, <=3.10.0",
        "numpy>=1.21.6, <=2.1.2",
        "pandas>=1.3.5, <=2.2.3",
        "plotly>=5.18.0, <=5.24.1",
        "scikit-learn>=1.0.2, <=1.5.2",
        "shap>=0.41.0, <=0.46.0",
        "statsmodels>=0.12.2, <=0.14.4",
        "tqdm>=4.66.4, <=4.67.1",
    ],
)
