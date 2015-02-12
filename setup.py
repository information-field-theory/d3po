## D3PO (Deonoiseing, Deconvolving, and Decomposing Photon Observations) has
## been developed at the Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2014 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/d3po/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup

setup(name="ift_d3po",
      version="1.0.1",
      author="Marco Selig",
      author_email="mselig@mpa-garching.mpg.de",
      #maintainer="",
      #maintainer_email="@mpa-garching.mpg.de",
      description="Deonoiseing, Deconvolving, and Decomposing Photon Observations",
      url="http://www.mpa-garching.mpg.de/ift/d3po/",
      packages=["d3po"],
      package_dir={"d3po": "."},
      package_data={"d3po": ["config",
                             "demo_config",
                             "demo_kernel.txt",
                             "demo_events.txt",
                             "demo_exposure.txt",
                             "demo_mask.txt"]},
      license="GPLv3")

