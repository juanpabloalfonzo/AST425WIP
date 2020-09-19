from marvin.tools.maps import Maps
from marvin.tools.cube import Cube
import matplotlib.pyplot as plt


maps = Maps(plateifu='8714-12703')
print(maps)
# get an emission line map
haflux = maps.emline_gflux_ha_6564
values = haflux.value
ivar = haflux.ivar
mask = haflux.mask
haflux.plot()

cube = Cube(plateifu='8714-12703')
# get a spaxel by slicing cube[i,j]
spaxel=cube[16, 16]
flux = spaxel.flux
wave = flux.wavelength
ivar = flux.ivar
mask = flux.mask
flux.plot()

#plt.show()