from setuptools import Command, Extension, find_packages, setup
from pkg_resources import parse_requirements as _parse_requirements

#from sksurv setuptools



def parse_requirements(filename):
    with open(filename) as fin:
        parsed_requirements = _parse_requirements(
            fin)
        requirements = [str(ir) for ir in parsed_requirements]
    return requirements

def get_extensions():
    import numpy

    numpy_includes = [numpy.get_include()]
    _check_eigen_source()

    extensions = []
    for config in EXTENSIONS.values():
        name = get_module_from_sources(config["sources"])
        include_dirs = numpy_includes + config.get("include_dirs", [])
        extra_compile_args = config.get("extra_compile_args", [])
        language = config.get("language", "c")
        ext = Extension(
            name=name,
            sources=config["sources"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language=language,
        )
        extensions.append(ext)

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv and "clean" not in sys.argv:
        extensions = cythonize_extensions(extensions)

    return extensions

def setup_package():
    setup(
        name='xgbsurv',
        url='',
        project_urls={
            #"Bug Tracker": "",
            #"Documentation": "",
            "Source Code": "",
        },
        author='Julius Schulte',
        author_email='',
        description='Sklearn survival analysis with gradient boosted decision trees (GDBTs).',
        #long_description=get_long_description(),
        license="GPLv3+",
        packages=find_packages(),
        #ext_modules=get_extensions(),
        classifiers=['Development Status :: 1 - Planning',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                     'Operating System :: MacOS',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     #'Programming Language :: Python :: 3.9.5',
                     'Programming Language :: Python :: 3.10.9',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     ],
        zip_safe=False,
        #package_data={"datasets": ["data/*.arff"]},
        python_requires='>=3.8',
        install_requires=parse_requirements('requirements/requirements.txt'),
        #cmdclass={"clean": clean},
    )


if __name__ == "__main__":
    setup_package()
