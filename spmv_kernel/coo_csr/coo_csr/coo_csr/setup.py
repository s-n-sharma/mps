from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import subprocess


def compile_metal():
    print("Compiling coo_csr.metal...")
    os.system("xcrun -sdk macosx metal -c coo_csr.metal -o coo_csr.air")
    os.system("xcrun -sdk macosx metallib coo_csr.air -o shaders.metallib")
    if os.path.exists("coo_csr.air"):
        os.remove("coo_csr.air")

compile_metal()

current_dir = os.path.dirname(os.path.abspath(__file__))
metal_cpp_path = os.path.join(current_dir, "metal-cpp")

setup(
    name='coo_csr',
    ext_modules=[
        CppExtension(
            name='coo_csr',
            sources=['coo_to_csr.cpp'],
            include_dirs=[metal_cpp_path],
            extra_compile_args=['-std=c++17'],
            extra_link_args=[
                '-framework', 'Metal', 
                '-framework', 'Foundation', 
                '-framework', 'QuartzCore'
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)