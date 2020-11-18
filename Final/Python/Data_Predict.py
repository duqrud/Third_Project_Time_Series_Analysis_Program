import sys
import os, UI4
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from keras.models import load_model
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.metrics import mean_squared_error
from math import sqrt

import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
import tensorflow as tf
from tensorflow import keras
import LS


class Data_Predict(QDialog, UI4.Ui_Dialog):
    def __init__(self, parent):
        super(Data_Predict, self).__init__(parent)
        self.setupUi(self)
        self.show()

        # 불러올 파일 초기화
        # 다이어로그 크기
        self.w = parent.w
        self.h = parent.h
        # self.c = parent.c
        self.setFixedSize(self.w, self.h)
        self.center()

        # LSTM 콤보 박스에 데이터 넣기
        self.LSTMcomboboxes = [
            self.Date_combobox,
            self.Open_combobox,
            self.High_combobox,
            self.Low_combobox,
            self.Close_combobox,
            self.Volume_combobox
        ]
        # 그래프

        # plt.style.use(color='#19232D')
        self.fig = plt.Figure()
        self.fig.set_facecolor("None")
        self.canvas = FigureCanvas(self.fig)
        self.result_gridLayout.addWidget(self.canvas)
        self.fig2 = plt.Figure()
        self.fig2.set_facecolor("None")
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.hide()
        self.result_gridLayout.addWidget(self.canvas2)
        self.fig3 = plt.Figure()
        self.fig3.set_facecolor("None")
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.hide()
        self.result_gridLayout.addWidget(self.canvas3)
        self.plus_pushButton.clicked.connect(self.plus_Btn_pushed)
        self.minus_pushButton.clicked.connect(self.minus_Btn_pushed)
        
        self.predict_periods.setValidator(QIntValidator(self))

        # 모델 불러오기
        self.data = 'x'
        self.model_fit = 'x'
        self.model_status = 'x'
        self.show()

        self.Import_Data.setStyleSheet(
            'image:url(./dataset2.webp); background-color:rgb(25,35,45); border-width:5px; ')
        self.Import_Model.setStyleSheet(
            'image:url(./model1.webp); background-color:rgb(25,35,45); border-width:5px;')

        self.predict_pushbutton.setStyleSheet(
            'image:url(./prediction7.png); background-color:rgb(25,35,45); border-width:5px;'
        )

        # pushbutton
        self.Import_Data.setCursor(QCursor(Qt.PointingHandCursor))
        self.Import_Data.clicked.connect(self.Open_File)
        self.Import_Model.setCursor(QCursor(Qt.PointingHandCursor))
        self.Import_Model.clicked.connect(self.showModel)
        self.predict_pushbutton.setCursor(QCursor(Qt.PointingHandCursor))
        self.predict_pushbutton.clicked.connect(self.predict_start)

        # 데이터 불러오기 및 그래프 레이아웃
        # self.start_prediction.setPlaceholderText("값을 입력해 주세요.")
        # self.setFocus()

        # 그래프 크기 
        self.groupBox_3.setMinimumWidth(self.w // 2 * 0.9)

        # 불러온 모델 학습
        # self.prediction.clicked.connect(self.predict)

        # 파일 저장
        self.save_csv.clicked.connect(self.csv_save)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def Open_File(self):
        self.showFile()
        if self.filename[0] == '':
            return
        # self.start_prediction.setPlaceholderText(
        #     'ex) ' + str(self.data.iloc[self.r - 1, 0]))

    def showFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        self.filename = QFileDialog.getOpenFileName(
            self, "파일열기", "", "(*.csv);;(*.json)")

        if self.filename[0] == '':
            return
        if self.filename[0][-1] == 'v':
            try:
                self.data = pd.read_csv(
                    self.filename[0], squeeze=True)
                # print('1')
                # print(self.filename)
                self.data_name1 = self.filename[0][::-1]
                self.res1_name = ''
                for i in self.data_name1:
                    if i == '/':
                        break
                    else:
                        self.res1_name = i + self.res1_name
                self.data_name.setText(self.res1_name)
                try:
                    if int(self.data.columns[0]) or float(self.data.columns[0]):
                        c = self.data.shape[1]
                        head = [str(i)+'열' for i in range(1, c + 1)]
                        self.data = pd.read_csv(
                            self.filename[0], squeeze=True, names=head, encoding='cp949'
                        )
                except:
                    pass
            except:
                self.data = pd.read_csv(
                    self.filename[0], squeeze=True, encoding='cp949')
                c = self.data.shape[1]
                head = [str(i)+'열' for i in range(1, c + 1)]
                self.data = pd.read_csv(
                    self.filename[0], squeeze=True, names=head, encoding='cp949'
                )

        else:
            self.data = pd.read_json(
                self.filename[0], squeeze=True)
        self.comboBox_x.clear()
        self.comboBox_y.clear()
        for i in self.data.columns:
            self.comboBox_x.addItem(i)
            self.comboBox_y.addItem(i)
        # print(type(self.data[self.comboBox_x.currentText()][0]))
        if type(self.data[self.comboBox_x.currentText()][0]) == str:
            try:
                self.data[self.comboBox_x.currentText()] = pd.to_datetime(self.data[self.comboBox_x.currentText()], format='%Y-%m-%d')
            except:
                print('잘못왔어요')
                pass
        print(type(self.data[self.comboBox_x.currentText()][0]))
        self.c = len(self.data.columns)
        self.r = len(self.data.index)
        self.data_file = self.data.columns
        self.data_load = self.data

        self.LSTM_setupUI()
        return self.showTable()

    def showModel(self):
        self.load_file = QFileDialog.getOpenFileName(
            self, '파일열기', '', '(*.pkl *.h5);; (*.pkl);; (*.h5)')
        if self.load_file[0] == '':
            return
        try:
            if self.load_file[0][-3:] == 'pkl':
                self.tabWidget.setCurrentIndex(0)
                self.model = ARIMAResults.load(self.load_file[0])
                self.name = self.load_file[0][::-1]
                self.res_name = ''
                for i in self.name:
                    if i == '/':
                        break
                    else:
                        self.res_name = i + self.res_name
            else:
                self.model = load_model(load_file[0])
        except:
            print(self.load_file)
            if self.load_file[0][-2:] =='h5':
                self.tabWidget.setCurrentIndex(1)
                self.model = tf.keras.models.load_model(self.load_file[0])
                self.name = self.load_file[0][::-1]
                self.res_name = ''
                for i in self.name:
                    if i == '/':
                        break
                    else:
                        self.res_name = i + self.res_name
                # print(self.model)
                # Model_name = Model_name + '.h5'
                # model = tf.keras.models.load_model(Model_path + '/' + Model_name)
        QMessageBox.about(self, "Alert", "성공적으로 불러왔습니다.")
        self.model_name.setText(self.res_name)

    def showTable(self):
        self.tableWidget.setRowCount(self.r)
        self.tableWidget.setColumnCount(self.c)
        self.tableWidget.setHorizontalHeaderLabels(self.data.columns)
        
        for i in range(self.r):
            for j in range(self.c):
                if type(self.data.iloc[i, j]) == pd._libs.tslibs.timestamps.Timestamp:
                    self.tableWidget.setItem(
                        i, j, QTableWidgetItem(self.data.iloc[i, j].to_pydatetime().strftime('%Y-%m-%d')))
                else:
                    self.tableWidget.setItem(
                        i, j, QTableWidgetItem(str(self.data.iloc[i, j])))
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)


    def predict_start(self):
        try:
            if self.data == 'x':
                QMessageBox.about(self, "Alert", "데이터를 불러와 주세요.")
                return
        except:
            pass
        try:
            if self.model == 'x':
                QMessageBox.about(self, "Alert", "모델을 불러와 주세요.")
                return
        except:
            pass
        self.p = self.predict_periods.text()
        if self.p == '':
            QMessageBox.about(self, "Alert", "예측할 기간을 입력해 주세요.")
            return
        
        if self.res_name[-3:] == 'pkl':
            if self.comboBox_x.currentText() == self.comboBox_y.currentText():
                QMessageBox.about(self, "Alert", "x축 y축을 다르게 설정해주세요.")
                return
            if self.predict_periods.text() == '':
                QMessageBox.about(self, "Alert", "기간을 설정해주세요.")
                return
            self.fig.clear()
            self.fig2.clear()
            self.fig3.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            self.fig.set_facecolor("white")
            ax2 = self.fig2.add_subplot(1, 1, 1)
            self.fig2.set_facecolor("white")
            ax3 = self.fig3.add_subplot(1, 1, 1)
            self.fig3.set_facecolor("white")

            series_data = pd.Series(list(self.data[self.comboBox_y.currentText()]), index=self.data[self.comboBox_x.currentText()])
            self.model = self.model.apply(series_data, refit=True )
            self.model_fit = self.model.model.fit()
            self.model.summary()

            self.fore = self.model_fit.forecast(steps=int(self.p))
            self.idx = self.fore.index
            
            ax.plot(series_data, color='blue', label='actual')
            ax.plot(self.fore, color='red', label='predict')
            ax.set_xlabel(self.comboBox_x.currentText())
            ax.set_ylabel(self.comboBox_y.currentText())
            ax.legend(loc="upper left")
            ax.grid()
            self.canvas.draw()
            ax2.plot(self.fore, color='red', label='predict')
            ax2.set_xlabel(self.comboBox_x.currentText())
            ax2.set_ylabel(self.comboBox_y.currentText())
            ax2.legend(loc="upper left")
            ax2.grid()
            self.canvas2.draw()

        elif self.res_name[-2:] == 'h5':
            self.LSTM_Test()
        
        num_rows = self.tableWidget.rowCount()
        num_cols = self.tableWidget.columnCount()

        col = [self.comboBox_x.currentText(), self.comboBox_y.currentText()]

        self.tmp_df = pd.DataFrame(
            columns=col, index=range(num_rows))

        try:
            if int(self.data_load.iloc[self.r - 1, 0]):
                self.tableWidget.clear()
                self.tableWidget.setRowCount(int(self.p))
                self.tableWidget.setColumnCount(2)
                r = 0
                self.tableWidget.setHorizontalHeaderLabels([self.comboBox_x.currentText(), self.comboBox_y.currentText()])
                for i in range(int(self.data_load.iloc[self.r - 1, 0]) + 1, int(self.data_load.iloc[self.r - 1, 0]) + int(self.p)+1):
                    self.tableWidget.setItem(
                        r, 0, QTableWidgetItem(str(i)))
                    self.tableWidget.setItem(
                        r, 1,  QTableWidgetItem(str(self.fore[self.idx[r]])))
                    self.tmp_df.iloc[r, 0] = str(i)
                    self.tmp_df.iloc[r, 1] = self.fore[self.idx[r]]
                    r += 1
                c = 0
                print(self.tmp_df)
                print(type(self.tmp_df))
                self.tableWidget.setEditTriggers(
                    QAbstractItemView.NoEditTriggers)

        except:
            if pd.to_datetime(self.data_load.iloc[self.r - 1, 0], format='%Y-%m-%d'):
                self.date_time = pd.date_range(self.data_load.iloc[self.r - 1, 0], periods=int(self.p) + 1, freq='D')
            print(pd.date_range(self.data_load.iloc[self.r - 1, 0], periods=int(self.p) + 1, freq='D'))
        
            self.tableWidget.clear()

            self.tmp_df = pd.DataFrame(
                columns=col, index=range(int(self.p)))    

            self.tableWidget.setRowCount(int(self.p))
            self.tableWidget.setColumnCount(2)

            r = 0
            self.tableWidget.setHorizontalHeaderLabels([self.comboBox_x.currentText(), self.comboBox_y.currentText()])

            for i in self.date_time[1:]:
                self.tableWidget.setItem(
                    r, 0, QTableWidgetItem(str(i)))
                self.tableWidget.setItem(
                    r, 1,  QTableWidgetItem(str(self.fore[self.idx[r]])))
                self.tmp_df.iloc[r, 0] = str(i)
                self.tmp_df.iloc[r, 1] = self.fore[self.idx[r]]
                r += 1
            c = 0
            self.tableWidget.setEditTriggers(
                QAbstractItemView.NoEditTriggers)
    def csv_save(self):
        try:
            csv = self.tmp_df
        except:
            QMessageBox.about(self, "Alert", "먼저 데이터와 모델을 불러와 예측을 해주세요.")
            return
        savefile = QFileDialog.getSaveFileName(self, '파일저장', '', '(*.csv)')
        csv.to_csv(savefile[0], index=False)

    def plus_Btn_pushed(self):
        if self.tabWidget.indexOf(self.tab_7) == 0:
            self.canvas.hide()
            self.canvas2.show()
        else:
            pass

    def minus_Btn_pushed(self):
        if self.tabWidget.indexOf(self.tab_7) == 0:
            self.canvas2.hide()
            self.canvas.show()
        else:
            pass
    
    def LSTM_setupUI(self):
        
        for combo in self.LSTMcomboboxes:
            combo.clear()
            for idx in range(self.c):
                combo.addItem(self.data.columns[idx])

    def LSTM_Test(self):
        self.canvas.hide()
        self.fig3.clear()
        stock_file = self.data.loc[:, [self.Date_combobox.currentText()]]
        for combo in self.LSTMcomboboxes:
            if combo == self.Date_combobox:
                continue
            stock_file = pd.merge(
                stock_file, self.data.loc[:, [combo.currentText()]], left_index=True, right_index=True, how='left')
        # window_size = int(self.Window_size_lineText.text())
        dense = int(self.predict_periods.text())
        graph, result = LS.Test(stock_file, 0.95, dense, dense, self.model)
        self.fore = result
        self.idx = range(len(self.fore))

        try:
            self.result_gridLayout.removeWidget(self.canvas3)
        except:
            pass
        self.canvas3 = FigureCanvas(graph)
        self.result_gridLayout.addWidget(self.canvas3)
        self.canvas3.show()
        self.minus_pushButton.hide()
        self.plus_pushButton.hide()