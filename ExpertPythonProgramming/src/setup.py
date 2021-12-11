from setuptools import setup, Extension

setup(
    name='fibonacci',
    ext_modules=[
        Extension('fibonacci', ['test_c_fibonacci.cpp'])
    ]
)
