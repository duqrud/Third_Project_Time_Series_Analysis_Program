import sys, UI2

import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
import Train_Dialog
import matplotlib.pyplot as plt
import missingno as msno
import matplotlib
from matplotlib import font_manager, rc
matplotlib.rcParams['axes.unicode_minus'] = False
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class new_Dialog(QDialog, UI2.Ui_Data_Dialog):
    def __init__(self, parent):
        super(new_Dialog, self).__init__(parent)
        self.setupUi(self)
        self.w = parent.w
        self.h = parent.h
        self.setFixedSize(self.w, self.h)
        self.data = parent.series_1
        self.r = parent.r
        self.c = parent.c
        self.colum = self.c
        self.colum_arr = [x for x in range(self.c)]
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.graph_view.addWidget(self.canvas)
        self.graph_frame.addLayout(self.graph_view)
        self.setLayout(self.graph_frame)

        self.setupUI(self.data, self.r, self.c)

        self.show()

    def setupUI(self, data, r, c):
        self.origin_data = self.data
        self.origin_col = self.c
        self.changeheader()

        self.Save_Button.clicked.connect(self.saveFunction)
        self.Training_Button.clicked.connect(self.training_btn)

        self.formLayout_3 = QFormLayout(self.scrollAreaWidgetContents)
        self.formLayout_3.setObjectName("formLayout_3")
        for i in range(c):
            self.checkbox = QCheckBox(self.scrollAreaWidgetContents)
            self.checkbox.setMinimumSize(QSize(0, 30))
            self.checkbox.setObjectName(f"checkbox_{i}")
            self.checkbox.setText(data.columns[i])
            self.formLayout_3.setWidget(i, QFormLayout.LabelRole, self.checkbox)

        self.del_Col.clicked.connect(self.del_columns)
        self.reset_Col.clicked.connect(lambda state, origin_data=self.origin_data: self.all_reset(state, self.origin_data, self.origin_col))

        self.draw_graph()

        


        # 결측치 처리
        self.text = self.data.isnull().sum()
        self.formLayout_2 = QFormLayout(self.scrollAreaWidgetContents_3)
        self.formLayout_2.setObjectName("formLayout_2")
        for i in range(c):
            self.label = QLabel(self.scrollAreaWidgetContents_3)
            self.label.setMinimumSize(QSize(0, 30))
            self.label.setObjectName(f'label_{i}')
            self.formLayout_2.setWidget(i, QFormLayout.LabelRole, self.label)
            self.label.setText(data.columns[i])

            self.combo = QComboBox(self.scrollAreaWidgetContents_3)
            self.combo.setMinimumHeight(30)
            self.combo.setObjectName(f'comboBox_{i}')
            self.combo.addItem('제거')
            self.combo.addItem('0으로 대체')
            self.combo.addItem('평균값으로 대체')
            self.combo.addItem('중앙값으로 대체')
            self.formLayout_2.setWidget(i, QFormLayout.FieldRole, self.combo)
        self.radioButton.setChecked(True)
        self.process_Button.clicked.connect(lambda state, origin_col=self.origin_col: self.process_btn(state, self.origin_col))
        self.reset_Button.clicked.connect(
            lambda state, origin_data=self.origin_data: self.reset_btn(state, self.origin_data))

        column_headers = data.columns

        self.tableWidget.resize(500, 300)
        if r >= 200:
            r = 200
        self.tableWidget.setRowCount(r)
        self.tableWidget.setColumnCount(c)
        self.tableWidget.setHorizontalHeaderLabels(column_headers)

        for i in range(r):
            for j in range(c):
                self.tableWidget.setItem(
                    i, j, QTableWidgetItem(str(data.iloc[i, j])))
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.groupBox.setMinimumWidth(self.w // 3 * 0.9)
        self.groupBox_2.setMinimumWidth(self.w // 3 * 0.9)
        self.groupBox_3.setMinimumWidth(self.w // 3 * 0.9)
        self.groupBox_4.setMinimumWidth(self.w // 2 * 0.9)
        self.groupBox_5.setMinimumWidth(self.w // 2 * 0.9)

    def changeheader(self):
        # 컬럼명 변경
        self.formLayout = QFormLayout(self.scrollAreaWidgetContents_2)
        self.formLayout.setObjectName("formLayout")
        for idx in range(self.c):
            self.label = QLabel(self.scrollAreaWidgetContents_2)
            self.label.setMinimumSize(QSize(0, 30))

            self.label.setObjectName(f"label_{idx}")
            self.formLayout.setWidget(idx, QFormLayout.LabelRole, self.label)
            self.label.setText(self.data.columns[idx])

            self.lineEdit = QLineEdit(self.scrollAreaWidgetContents_2)
            self.lineEdit.setMinimumSize(QSize(0, 30))
            self.lineEdit.setObjectName(f"lineEdit_{idx}")
            self.formLayout.setWidget(
                idx, QFormLayout.FieldRole, self.lineEdit)
        self.pushButton = QPushButton('변경', self.groupBox_2)
        self.pushButton.setObjectName("changeButton")
        self.pushButton.clicked.connect(self.changeTextFunction)
        self.pushButton.setMaximumWidth(100)
        self.gridLayout_4.addWidget(self.pushButton, 2, 0, 1, 1, Qt.AlignRight)

    # 컬럼 header 변경
    def changeTextFunction(self):
        cnt = 0
        for i in self.colum_arr:
            try:
                text = self.findChild(QLineEdit, f"lineEdit_{i}").text()
                if text != '':
                    # self.data.columns.values[cnt] = text
                    self.data.rename(columns={self.data.columns.values[cnt]:text}, inplace=True)
                    self.findChild(QCheckBox, f"checkbox_{i}").setText(text)
                cnt += 1
                self.findChild(QLineEdit, f"lineEdit_{i}").clear()
            except:
                pass
        self.tableWidget.setHorizontalHeaderLabels(self.data.columns)

    # Tabel cell 내용 변경
    def changeTableCells(self, r, c):
        for i in range(r):
            for j in range(c):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.data.iloc[i, j])))
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

    # 전처리
    def process_btn(self, state, origin_col):
        if self.radioButton.isChecked():
            if self.all_comboBox.currentText() == '제거':
                self.data = self.data.dropna()
                self.r = self.data.shape[0]
                self.c = self.data.shape[1]
            elif self.all_comboBox.currentText() == '0으로 대체':
                self.data = self.data.fillna(0)
            elif self.all_comboBox.currentText() == '평균값으로 대체':
                self.data = round(self.data.fillna(self.data.mean()))
            elif self.all_comboBox.currentText() == '중앙값으로 대체':
                self.data = round(self.data.fillna(self.data.median()))

        else:
            n = 0
            for i in range(origin_col):
                combo = self.findChild(QComboBox, f"comboBox_{i}")
                if combo == None:
                    n += 1
                if combo != None:
                    combo = combo.currentText()
                    if combo == '제거':
                        self.data = self.data.dropna(subset=[self.data.columns[i - n]])
                        self.r = self.data.shape[0]
                    elif combo == '0으로 대체':
                        self.data[self.data.columns[i - n]] = self.data[self.data.columns[i - n]].fillna(0)
                    elif combo == '평균값으로 대체':
                        self.data[self.data.columns[i - n]] = self.data[self.data.columns[i - n]].fillna(
                            self.data[self.data.columns[i - n]].mean())
                    elif combo == '중앙값으로 대체':
                        self.data[self.data.columns[i - n]] = round(
                            self.data[self.data.columns[i - n]].fillna(self.data[self.data.columns[i - n]].median()))
        return self.changeTableCells(self.r, self.c), self.draw_graph()

    # graph 그리기
    def draw_graph(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)

        missing = self.data.isnull().sum()
        x_val = []
        y_val = []
        for i in range(self.data.shape[1]):
            cnt = self.data[self.data.columns[i]].isnull().sum()
            try:
                if cnt:
                    x_val.append(self.data.columns[i])
                    y_val.append(cnt)
                else:
                    x_val.append(self.data.columns[i])
                    y_val.append(0)
            except:
                pass

        try:
            missing = missing[missing > 0]
            missing.sort_values(inplace=True)
            ax.bar(x_val, y_val)
            ax.set_ylabel('개수')
            for i, v in enumerate(x_val):
                ax.text(v, y_val[i], str(y_val[i]))
            plt.bar(x_val, y_val)
            plt.close()
            

        except:
            ax.bar(x_val, y_val)
            ax.set_xlabel('Column')
            ax.set_ylabel('개수')
            for i, v in enumerate(x_val):
                ax.text(y_val[i], v, str(y_val[i]))

        ax.grid()
        self.canvas.draw()

    # column 삭제
    def del_columns(self):
        cnt = 0
        for i in range(self.colum):
            try:
                if self.findChild(QCheckBox, f"checkbox_{i}").isChecked():
                    col_name = self.findChild(QCheckBox, f"checkbox_{i}").text()
                    self.data = self.data.drop(col_name, axis=1)
                    cnt += 1
                    label = self.formLayout.labelForField(self.findChild(QLineEdit, f"lineEdit_{i}"))
                    label.deleteLater()
                    self.findChild(QLineEdit, f"lineEdit_{i}").deleteLater()
                    combo_label = self.formLayout_2.labelForField(self.findChild(QComboBox, f"comboBox_{i}"))
                    combo_label.deleteLater()
                    self.findChild(QComboBox, f"comboBox_{i}").deleteLater()
                    self.findChild(QCheckBox, f"checkbox_{i}").deleteLater()
                    self.colum_arr.remove(i)
            except:
                pass
        self.c -= cnt
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(self.data.columns)
        return self.changeTableCells(self.r, self.c), self.draw_graph()

    # 데이터 초기화
    def reset_btn(self, state, origin_data):
        self.data = origin_data
        self.colum = self.c
        self.colum_arr = [x for x in range(self.c)]
        return self.changeTableCells(self.r, self.c), self.draw_graph()

    # column 초기화
    def all_reset(self, state, origin_data, origin_col):
        self.data = origin_data
        self.c = origin_col
        self.colum = self.c
        self.colum_arr = [x for x in range(self.c)]
        for i in range(self.c):
            if self.findChild(QLabel, f"label_{i}") == None:
                self.label = QLabel(self.scrollAreaWidgetContents_2)
                self.label.setMinimumSize(QSize(0, 30))
                self.label.setObjectName(f"label_{i}")
                self.formLayout.setWidget(i, QFormLayout.LabelRole, self.label)
                self.label.setText(self.data.columns[i])
                self.lineEdit = QLineEdit(self.scrollAreaWidgetContents_2)
                self.lineEdit.setMinimumSize(QSize(0, 30))
                self.lineEdit.setObjectName(f"lineEdit_{i}")
                self.formLayout.setWidget(i, QFormLayout.FieldRole, self.lineEdit)

                self.label = QLabel(self.scrollAreaWidgetContents_3)
                self.label.setMinimumSize(QSize(0, 30))
                self.label.setObjectName(f'label_{i}')
                self.formLayout_2.setWidget(i, QFormLayout.LabelRole, self.label)
                self.label.setText(self.data.columns[i])

                self.combo = QComboBox(self.scrollAreaWidgetContents_3)
                self.combo.setMinimumHeight(30)
                self.combo.setObjectName(f'comboBox_{i}')
                self.combo.addItem('제거')
                self.combo.addItem('0으로 대체')
                self.combo.addItem('평균값으로 대체')
                self.combo.addItem('중앙값으로 대체')
                self.formLayout_2.setWidget(i, QFormLayout.FieldRole, self.combo)

                self.checkbox = QCheckBox(self.scrollAreaWidgetContents)
                self.checkbox.setMinimumSize(QSize(0, 30))
                self.checkbox.setObjectName(f"checkbox_{i}")
                self.checkbox.setText(self.data.columns[i])
                self.formLayout_3.setWidget(i, QFormLayout.LabelRole, self.checkbox)
        self.tableWidget.setHorizontalHeaderLabels(self.data.columns)
        return self.changeTableCells(self.r, self.c), self.draw_graph()

    # 데이터 저장
    def saveFunction(self):
        try:
            csv = self.data
            savefile = QFileDialog.getSaveFileName(self, '파일저장', '', '(*.csv)')
            csv.to_csv(savefile[0], index=False)
        except:
            return

    # training과 연동
    def training_btn(self):
        self.series_1 = self.data
        Train_Dialog.Train_Dialog(self)
        self.close()
