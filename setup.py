from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='magnet',
    version='0.9.3rc',
    description='PyTorch training manager (v0.9.3 Release Candidate 4)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kison Ho',
    author_email='unfit-gothic.0q@icloud.com',
    packages=[
        'magnet',
        'magnet.data',
        'magnet.managers',
        'magnet.nn',
        'torchmanager_monai',
        'torchmanager_monai.metrics',
    ],
    package_dir={
        'magnet': 'magnet',
        'magnet.data': 'magnet/data',
        'magnet.managers': 'magnet/managers',
        'magnet.networks': 'magnet/networks',
        'magnet.nn': 'magnet/nn',
        'torchmanager_monai': 'torchmanager_monai',
        'torchmanager_monai.metrics': 'torchmanager_monai/metrics',
    },
    install_requires=[
        'monai>=0.9.0',
        'torchmanager>=1.0.4',
        'torch>=1.8.2',
        'tqdm',
    ],
    python_requires=">=3.9",
    url="https://github.com/kisonho/magnet.git"
)
