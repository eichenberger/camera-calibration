from distutils.core import setup, Extension

module1 = Extension('orb',
                    sources = ['orbmodule.c'])

setup (name = 'Orb',
       version = '1.0',
       description = 'This is a orb package',
       ext_modules = [module1])
