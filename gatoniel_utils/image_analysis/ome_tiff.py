import numpy as np
import xml.etree.ElementTree as ET
from tifffile import TiffFile


def get_pixel_size(tif):
    root = ET.fromstring(tif.ome_metadata)
    schema = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
    namespace = root.attrib[schema].split(" ")[0]
    xml_pixels = root.findall(
        '{{{ns}}}Image/{{{ns}}}Pixels'.format(ns=namespace)
    )[0]
    res = float(xml_pixels.attrib['PhysicalSizeX'])
    return res


def get_time(tif):
    root = ET.fromstring(tif.ome_metadata)
    schema = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
    namespace = root.attrib[schema].split(" ")[0]
    xml_planes = root.findall(
        '{{{ns}}}Image/{{{ns}}}Pixels/{{{ns}}}Plane'.format(ns=namespace)
    )
    time = np.array([float(p.attrib["DeltaT"]) for p in xml_planes])
    time.sort()

    unit = xml_planes[0].attrib["DeltaTUnit"]

    return time, unit


def get_info_from_path(path):
    tif = TiffFile(path)
    res = get_pixel_size(tif)
    time, time_unit = get_time(tif)
    tif.close()

    return res, time, time_unit
