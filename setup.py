from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multisensor-sleep-analysis",
    version="0.1.0",
    author="MultiSensor Sleep Analysis Team",
    description="End-to-end pipeline for multi-sensor sleep analysis with PSG validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "multisensor-cli=scripts.cli:main",
            "multisensor-preprocess=scripts.preprocess_data:main",
            "multisensor-validate=scripts.validate_psg:main",
            "multisensor-visualize=scripts.visualize_results:main",
        ],
    },
)