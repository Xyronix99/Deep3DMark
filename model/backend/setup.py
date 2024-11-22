from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob
import os

_ext_src_root = "_ext"
_ext_sources  = glob.glob(f'{_ext_src_root}/src/*.cpp') + glob.glob(f'{_ext_src_root}/src/*.cu')

headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), _ext_src_root, 'include')

setup(
    name='backend',
    ext_modules=[
        CUDAExtension(
            name='backend._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", headers],
                "nvcc": ["-O2", headers]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)