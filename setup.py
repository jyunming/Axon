from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="local-rag-brain",
    version="2.0.0",
    author="Open Source Contributor",
    description="General-purpose open-source RAG and embedding interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "ollama>=0.1.0",
        "openai>=1.0.0",
        "rank_bm25>=0.2.2",
        "streamlit>=1.30.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "python-docx>=1.0.0",
        "pymupdf>=1.24.0",
        "pypdf>=4.0.0",
    ],
    extras_require={
        "qdrant": ["qdrant-client>=1.7.0"],
        "fastembed": ["fastembed>=0.1.0"],
        "all": [
            "qdrant-client>=1.7.0",
            "fastembed>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-brain=rag_brain.main:main",
            "rag-brain-ui=rag_brain.webapp:main_ui",
            "rag-brain-api=rag_brain.api:main",
        ],
    },
)
