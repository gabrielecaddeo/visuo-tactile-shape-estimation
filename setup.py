from setuptools import setup

setup(
    name='surface_classifier',
    version= '0.1.0',
    author = 'Gabriele Mario Caddeo',
    author_email = 'gabriele.caddeo@iit.it',
    install_requires = [
        'digit-interface',
        'opencv-python',
        'torch',
        'torchsummary',
        'torchvision',
        'vtk'
    ],
    packages = ['surfaceclassifier'],

)