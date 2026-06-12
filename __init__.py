from .plugin import STCRCPlugin


def classFactory(iface):
    return STCRCPlugin(iface)
