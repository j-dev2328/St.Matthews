import sys
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)
    from main_ui import MainUI
    window = MainUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
