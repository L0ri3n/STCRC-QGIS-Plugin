import os
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QIcon
from .dialog import STCRCDialog


class STCRCPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        self.action = QAction(QIcon(icon_path), "STCRC: Change Regime Classification", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&STCRC", self.action)

    def unload(self):
        self.iface.removePluginRasterMenu("&STCRC", self.action)
        self.iface.removeToolBarIcon(self.action)
        del self.action

    def run(self):
        dlg = STCRCDialog(self.iface)
        dlg.exec_()
