from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RagMedicalChatbot",
    version="0.1",
    author="mohamad_gamal",
    packages=find_packages(),
    install_requires = requirements,
)