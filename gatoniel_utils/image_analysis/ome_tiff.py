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
