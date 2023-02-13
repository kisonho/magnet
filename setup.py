from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='magnet',
    version='1.1',
    description='PyTorch training manager (v1.1)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kison Ho',
    author_email='unfit-gothic.0q@icloud.com',
    packages=[
        'magnet',
        'magnet.data',
        'magnet.losses',
        'magnet.managers',
        'magnet.networks',
        'magnet.nn',
        'torchmanager_monai',
        'torchmanager_monai.metrics',
    ],
    package_dir={
        'magnet': 'magnet',
        'magnet.data': 'magnet/data',
        'magnet.losses': 'magnet/losses',
        'magnet.managers': 'magnet/managers',
        'magnet.networks': 'magnet/networks',
        'magnet.nn': 'magnet/nn',
        'torchmanager_monai': 'torchmanager_monai',
        'torchmanager_monai.metrics': 'torchmanager_monai/metrics',
    },
    install_requires=[
        'monai>=0.9.0',
        'torch>=1.12.1',
        'torchmanager>=1.0.4',
        'tqdm',
    ],
    python_requires=">=3.9",
    url="https://github.com/kisonho/magnet.git"
)
