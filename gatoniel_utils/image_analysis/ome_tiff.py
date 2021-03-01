import numpy as np
import xml.etree.ElementTree as ET


def get_resolution(tif):
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
    time = np.array([p.attrib["DeltaT"] for p in xml_planes])
    time.sort()

    unit = xml_planes[0].attrib["DeltaTUnit"]

    return time, unit
