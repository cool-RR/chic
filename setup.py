import setuptools
import re
import pathlib

def read_file(name: str) -> str:
    return pathlib.Path(name).read_text(encoding='utf-8')


version = re.search(r"__version__ = '([0-9.]*)'", read_file('chic/__init__.py')).group(1)

requirements = read_file('requirements.txt')


setuptools.setup(
    name='chic',
    version=version,
    author='Ram Rachum',
    author_email='ram@rachum.com',
    description='Chic',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/cool-RR/chic',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.12',
    entry_points={
        'console_scripts': [
            'chic = chic.cling:cli'
        ],
    },
    extras_require={
        'tests': {
            'pytest',
            'pytest-xdist',
            'pytest-html',
        },
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
