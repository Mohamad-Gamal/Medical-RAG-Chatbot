from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RagMedicalChatbot",
    version="0.1.0",
    author="Mohamad Gamal",
    description="Medical RAG Chatbot with FAISS and HuggingFace embeddings",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,  # include non-Python files if needed
    entry_points={  # optional: for creating CLI commands
        "console_scripts": [
            "run-rag-chatbot=app.components.run_pipeline:rebuild_vector_store",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
