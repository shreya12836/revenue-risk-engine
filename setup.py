from setuptools import setup, find_packages

setup(
    name="revenue-risk-engine",
    version="0.1.0",
    description="Production customer churn and revenue-at-risk prediction system",
    author="Shreya Mishra",
    author_email="your-email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "shap>=0.42",
        "streamlit>=1.28",
        "imbalanced-learn>=0.11",
        "optuna>=3.4",
        "httpx>=0.25",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.6",
        ]
    },
)