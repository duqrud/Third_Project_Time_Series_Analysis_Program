# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '../_uiFiles/Main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SAVEBtn(object):
    def setupUi(self, SAVEBtn):
        SAVEBtn.setObjectName("SAVEBtn")
        SAVEBtn.resize(1086, 839)
        SAVEBtn.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(SAVEBtn)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        self.Test_pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Test_pushButton.sizePolicy().hasHeightForWidth())
        self.Test_pushButton.setSizePolicy(sizePolicy)
        self.Test_pushButton.setMinimumSize(QtCore.QSize(300, 300))
        self.Test_pushButton.setMaximumSize(QtCore.QSize(450, 450))
        self.Test_pushButton.setText("")
        self.Test_pushButton.setObjectName("Test_pushButton")
        self.gridLayout_2.addWidget(self.Test_pushButton, 1, 6, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 7, 1, 1)
        self.Dataset_pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Dataset_pushButton.sizePolicy().hasHeightForWidth())
        self.Dataset_pushButton.setSizePolicy(sizePolicy)
        self.Dataset_pushButton.setMinimumSize(QtCore.QSize(300, 300))
        self.Dataset_pushButton.setMaximumSize(QtCore.QSize(450, 450))
        self.Dataset_pushButton.setText("")
        self.Dataset_pushButton.setObjectName("Dataset_pushButton")
        self.gridLayout_2.addWidget(self.Dataset_pushButton, 1, 1, 1, 1)
        self.Training_pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Training_pushButton.sizePolicy().hasHeightForWidth())
        self.Training_pushButton.setSizePolicy(sizePolicy)
        self.Training_pushButton.setMinimumSize(QtCore.QSize(300, 300))
        self.Training_pushButton.setMaximumSize(QtCore.QSize(450, 450))
        self.Training_pushButton.setText("")
        self.Training_pushButton.setObjectName("Training_pushButton")
        self.gridLayout_2.addWidget(self.Training_pushButton, 1, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 3, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 6, 1, 1, QtCore.Qt.AlignHCenter)
        self.arrow1 = QtWidgets.QLabel(self.centralwidget)
        self.arrow1.setMinimumSize(QtCore.QSize(40, 40))
        self.arrow1.setMaximumSize(QtCore.QSize(40, 40))
        self.arrow1.setText("")
        self.arrow1.setObjectName("arrow1")
        self.gridLayout_2.addWidget(self.arrow1, 1, 2, 1, 1)
        self.arrow2 = QtWidgets.QLabel(self.centralwidget)
        self.arrow2.setMinimumSize(QtCore.QSize(40, 40))
        self.arrow2.setMaximumSize(QtCore.QSize(40, 40))
        self.arrow2.setText("")
        self.arrow2.setObjectName("arrow2")
        self.gridLayout_2.addWidget(self.arrow2, 1, 4, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        SAVEBtn.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(SAVEBtn)
        self.statusbar.setObjectName("statusbar")
        SAVEBtn.setStatusBar(self.statusbar)
        self.OpenBtn = QtWidgets.QAction(SAVEBtn)
        self.OpenBtn.setCheckable(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/folder.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.OpenBtn.setIcon(icon)
        self.OpenBtn.setObjectName("OpenBtn")
        self.SAVE = QtWidgets.QAction(SAVEBtn)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SAVE.setIcon(icon1)
        self.SAVE.setObjectName("SAVE")
        self.CloseBtn = QtWidgets.QAction(SAVEBtn)
        self.CloseBtn.setObjectName("CloseBtn")
        self.Import_modelBtn = QtWidgets.QAction(SAVEBtn)
        self.Import_modelBtn.setObjectName("Import_modelBtn")
        self.TrainBtn = QtWidgets.QAction(SAVEBtn)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/Training.webp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.TrainBtn.setIcon(icon2)
        self.TrainBtn.setObjectName("TrainBtn")

        self.retranslateUi(SAVEBtn)
        QtCore.QMetaObject.connectSlotsByName(SAVEBtn)

    def retranslateUi(self, SAVEBtn):
        _translate = QtCore.QCoreApplication.translate
        SAVEBtn.setWindowTitle(_translate("SAVEBtn", "MainWindow"))
        self.label.setToolTip(_translate("SAVEBtn", "<html><head/><body><p>분석하고 싶은 데이터를 로드하여 전처리 창으로 넘어갑니다.</p></body></html>"))
        self.label.setText(_translate("SAVEBtn", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">Load Data Set</span></p></body></html>"))
        self.label_2.setToolTip(_translate("SAVEBtn", "<html><head/><body><p>불러온 데이터를 학습합니다.</p></body></html>"))
        self.label_2.setText(_translate("SAVEBtn", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">Training</span></p></body></html>"))
        self.label_3.setToolTip(_translate("SAVEBtn", "<html><head/><body><p>데이터와 모델을 불러와서 예측합니다.</p></body></html>"))
        self.label_3.setText(_translate("SAVEBtn", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">Predict</span></p></body></html>"))
        self.OpenBtn.setText(_translate("SAVEBtn", "Open"))
        self.OpenBtn.setShortcut(_translate("SAVEBtn", "Ctrl+O"))
        self.SAVE.setText(_translate("SAVEBtn", "Save"))
        self.SAVE.setShortcut(_translate("SAVEBtn", "Ctrl+S"))
        self.CloseBtn.setText(_translate("SAVEBtn", "Close"))
        self.CloseBtn.setShortcut(_translate("SAVEBtn", "Ctrl+Q"))
        self.Import_modelBtn.setText(_translate("SAVEBtn", "Import model"))
        self.Import_modelBtn.setShortcut(_translate("SAVEBtn", "Ctrl+M"))
        self.TrainBtn.setText(_translate("SAVEBtn", "Train"))
        self.TrainBtn.setShortcut(_translate("SAVEBtn", "Ctrl+T"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SAVEBtn = QtWidgets.QMainWindow()
    ui = Ui_SAVEBtn()
    ui.setupUi(SAVEBtn)
    SAVEBtn.show()
    sys.exit(app.exec_())
