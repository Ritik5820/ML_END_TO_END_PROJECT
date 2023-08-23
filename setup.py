from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list:
    '''
    Function will return the list of requirements
    '''
    with open(file_path) as f:
        requirements = f.read().splitlines()

    # Remove "-e ." requirement if present
    requirements = [req for req in requirements if req != "-e ."]

    return requirements

setup(
    name="MLproject",
    version="0.0.1",
    author="Ritik",
    author_email="kohadritik@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
