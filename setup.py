from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open-coscientist-agents",
    version="0.0.1",
    author="conradry",
    author_email="",  # Add your email if you want to include it
    description="Implementation of multi-agent system for AI co-scientist",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conradry/open-coscientist-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.3.25",
        "langchain-community>=0.3.24",
        "langgraph>=0.4.7",
        "typing-extensions>=4.0.0",
        "ipython>=8.0.0",  # For notebook support
        "gpt-researcher>=0.15.1",  # deep-mode web research; 0.15+ = langchain 0.3 compatible
        "langchain-core>=0.3.65",
        "langchain-community>=0.3.2",
        # All chat models are routed through RouterAI's OpenAI-compatible API
        # and embeddings through an API endpoint via langchain-openai; the
        # per-provider SDKs (langchain-anthropic / langchain-google-genai) are no
        # longer imported by the core and are intentionally not required here.
        "langchain-openai>=0.3.18",
        "networkx>=3.5",
        "scikit-learn>=1.7.0",
        # Private-corpus ingestion (Phase 2) and export (Phase 4).
        "numpy>=1.24.0,<2.3",  # scipy (via scikit-learn) requires numpy<2.3
        "pymupdf>=1.24.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.0",
        "reportlab>=4.0.0",
        "markdown>=3.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "ruff>=0.0.1",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
