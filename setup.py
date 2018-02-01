from setuptools import setup

setup(name='halomodel',
      version='1.0.2',
      description='Library for halo model functions',
      url='https://github.com/jcoupon/halomodel',
      author='Jean coupon',
      author_email='jean.coupon@gmail.com',
      license='MIT',
      packages=['halomodel'],
      install_requires=[
        'numpy', 'astropy', 'scipy'
      ],
      zip_safe=False)
