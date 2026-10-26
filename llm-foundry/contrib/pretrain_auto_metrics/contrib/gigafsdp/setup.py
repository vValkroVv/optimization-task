from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)


def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with req_path.open("r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f.read().splitlines()
                if not line.startswith("#") and line.strip()
            ]
    return []


setup(
    name="gigafsdp",
    version="0.1.0",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["giga_fsdp", "giga_fsdp.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "test": [
            "pytest>=8.0.0",
            "transformers>=4.50.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
