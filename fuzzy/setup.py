from os.path import join, splitext

import numpy as np

try:
      from setuptools import setup, Extension
      from setuptools.command.build_ext import build_ext
except ImportError:
      from distutils.core import setup, Extension
      from distutils.command.build_ext import build_ext


PACKAGE_DIR = 'fuzzy'

ext_module = Extension('fuzzy_ext', 
                       language='cuda',
                       include_dirs=[np.get_include()],
                       sources=[join(PACKAGE_DIR, 'ext', file) for file in ['fuzzy_wrapper.c', 'fuzzy_cpu.c', 'fuzzy_gpu.cu']],
                       extra_compile_args={
                             'gcc': ['-O0'],
                             'nvcc': '-c -Xcompiler -fPIC,-g,-O0'.split()},
                       extra_link_args=['-lstdc++', '-L/usr/local/cuda/lib64', '-lcudart'])

def customize_compiler_for_cuda(compiler):
      compiler.src_extensions.append('.cu')
      default_compiler_so = compiler.compiler_so
      super = compiler._compile

      def _compile(self, src, ext, cc_args, extra_postargs, pp_opts):
            _, ext = splitext(src)
            if ext == '.cu':
                  compiler.set_executable('compiler_so', 'nvcc')
                  postargs = extra_postargs['nvcc']
            else:
                  postargs = extra_postargs['gcc']
            super(self, src, ext, cc_args, postargs, pp_opts)
            compiler.compiler_so = default_compiler_so

      compiler._compile = _compile

class CustomBuildExt(build_ext):
      def build_extensions(self):
            customize_compiler_for_cuda(self.compiler)
            build_ext.build_extensions(self)

setup(name='fuzzy',
      version='1.0',
      description='',
      package=['fuzzy'],
      ext_modules=[ext_module],
      package_dir={'fuzzy': PACKAGE_DIR},
      cmdclass={'build_ext': CustomBuildExt})
