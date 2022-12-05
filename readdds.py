from PIL import Image
import numpy as np
dds =Image.open("/home3/qinyiming/airender/data/Coc.dds")

data=np.array(dds)
print(data.shape)
print(data)


#from wand import image

#with image.Image("/home3/qinyiming/airender/data/Coc.dds") as img:
#    img.compression ='no'
#    print(img)

#import pyffi.formats.dds.DdsFormat
"""
import pyffi
from pyffi.formats.dds import DdsFormat
hex(DdsFormat.version_number('DX10'))
stream =open("/home3/qinyiming/airender/data/Coc.dds",'rb')
print(stream)
data =DdsFormat.Data()
data.inspect(stream)
print(data.header.pixel_format.size)
print(data.header.height)
data.read(stream)
#print(data.pixeldata.get_value())

for vnum in sorted(DdsFormat.versions.values()):
    print('0x%08X' % vnum)

"""