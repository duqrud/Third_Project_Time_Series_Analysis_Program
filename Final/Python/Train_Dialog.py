import sys, UI3
import pandas as pd
import PyQt5
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima.arima import auto_arima
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import LS

class Train_Dialog(QDialog, UI3.Ui_Train_Dialog):
    def __init__(self, parent):
        super(Train_Dialog, self).__init__(parent)
        self.setupUi(self)

        try:
            self.data = parent.series_1
        except:
            QMessageBox.about(self, "Alert", "분석하고 싶은 파일을 먼저 불러와주세요")
            return

        self.w = parent.w
        self.h = parent.h
        self.c = parent.c
        self.setFixedSize(parent.w, parent.h)

        self.LSTMcomboboxes = [
            self.Date_combobox,
            self.Open_combobox,
            self.High_combobox,
            self.Low_combobox,
            self.Close_combobox,
            self.Volume_combobox
        ]
        self.Train_pushButton.clicked.connect(self.LSTM_Train)
        self.LSTM_setupUI()



        self.parent = parent
        self.p_value = 0
        self.d_value = 0
        self.q_value = 0
        self.c_nc = 'c'
        self.day = 12
        self.model_status = 'x'
        self.fig = plt.Figure()
        self.fig.set_facecolor("none")
        self.fig2 = plt.Figure()
        self.fig2.set_facecolor("none")
        self.canvas = FigureCanvas(self.fig)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.hide()
        self.arima_gridLayout.addWidget(self.canvas)
        self.arima_gridLayout.addWidget(self.canvas2)
        self.plus_pushButton.clicked.connect(self.plus_Btn_pushed)
        self.minus_pushButton.clicked.connect(self.minus_Btn_pushed)
        self.plus_pushButton_2.clicked.connect(self.LSTM_plus_Btn_pushed)
        self.minus_pushButton_2.clicked.connect(self.LSTM_minus_Btn_pushed)
        self.Test_pushButton.clicked.connect(self.test_Btn_pushed)

        self.comboBox_x.clear()
        self.comboBox_y.clear()
        for i in self.data.columns:
            self.comboBox_x.addItem(i)
            self.comboBox_y.addItem(i)

        self.show()
        self.model_save_pushButton.clicked.connect(self.model_save_pushed)
        self.model_save_pushButton_2.clicked.connect(self.LSTM_model_save_pushed)
        self.progressBar.hide()
        self.Run_pushButton.clicked.connect(self.Run_Dialog)
        self.auto_arima_pushButton.clicked.connect(self.auto_arima_pushed)

        self.rmse_label.setMinimumWidth(self.w // 3 * 0.8)
        self.arima_groupBox.setMinimumWidth(self.w // 2 * 0.9)
        self.groupBox.setMinimumWidth(self.w // 2 * 0.9)

        self.groupBox_7.setMinimumWidth(self.w // 3 * 0.9)
        self.groupBox_8.setMinimumWidth(self.w // 3 * 0.9)
        self.groupBox_9.setMinimumWidth(self.w // 3 * 0.9)

        self.arima_groupBox_2.setMinimumWidth(self.w // 3 * 0.9)
        self.Loss_groupBox.setMinimumWidth(self.w // 3 * 0.9)
        self.Train_groupBox.setMinimumWidth(self.w // 3 * 0.9)

        self.studying_rate.setValidator(QIntValidator(self))
        self.verification_rate.setValidator(QIntValidator(self))
        self.test_rate.setValidator(QIntValidator(self))

        self.ModelSize_lineText.setValidator(QIntValidator(self))
        self.WindowSize_lineText.setValidator(QIntValidator(self))
        self.Epochs_linetext.setValidator(QIntValidator(self))
        self.BatchSize_lineText.setValidator(QIntValidator(self))
        self.Patience_lineText.setValidator(QIntValidator(self))
        self.PredictSize_lineText.setValidator(QIntValidator(self))

    def Run_Dialog(self):
        if self.studying_rate.text() == '':
            QMessageBox.about(self, "Alert", "학습 비율을 입력해주세요")
            return
        elif int(self.studying_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "학습 비율을 양수로 입력해주세요")
            return
        elif self.verification_rate.text() == '':
            QMessageBox.about(self, "Alert", "검증 비율을 입력해주세요")
            return
        elif int(self.verification_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "검증 비율을 양수로 입력해주세요")
            return
        elif self.test_rate.text() == '':
            QMessageBox.about(self, "Alert", "테스트 비율을 입력해주세요")
            return
        elif int(self.test_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "테스트 비율을 양수로 입력해주세요")
            return

        # Load한 파일의 행의 길이 측정 및 학습, 검증, 테스트 비율로 csv파일 길이 나눔
        # csv로 기준
        self.total_len = int(self.studying_rate.text()) + \
            int(self.verification_rate.text()) + int(self.test_rate.text())
        self.studying_rate_length = len(self.data) * \
            int(self.studying_rate.text()) // self.total_len
        self.verification_rate_length = len(
            self.data) * int(self.verification_rate.text()) // self.total_len

        # 분석 기법 선택
        if self.study_tabWidget.currentIndex() == 0:
            self.ARIMA_run(pd.Series(list(self.data[self.comboBox_y.currentText()]), index=self.data[self.comboBox_x.currentText()]))
        elif self.study_tabWidget.currentIndex() == 1:
            pass

    def ARIMA_run(self, series):
        if self.comboBox_x.currentText() == self.comboBox_y.currentText():
            QMessageBox.about(self, "Alert", "x축 y축을 다르게 설정해주세요.")
            return
        
        self.p_value = self.p_spinBox.value()
        self.d_value = self.d_spinBox.value()
        self.q_value = self.q_spinBox.value()
        self.rmse_label.clear()
        self.arimaresult_label_2.clear()
        self.fig.clear()
        self.fig2.clear()
        series.index = pd.to_datetime(series.index)
        
        try:
            series.index = pd.to_datetime(
                series.index)
            series.set_index(self.comboBox_x.currentText(), inplace=True)
            print('hello')
        except:
            pass
        series.astype('float32')
        months_in_year = 1
        series_studying = series[:self.studying_rate_length]
        series_verification = series[self.studying_rate_length:
                                self.studying_rate_length+self.verification_rate_length]
        self.series_train = series[:self.studying_rate_length+self.verification_rate_length]
        self.series_test = series[self.studying_rate_length +
                                  self.verification_rate_length:]
        self.len_test = len(self.series_test)
        len_arr = len(series_verification)
        self.progressBar.setMaximum(len_arr)
        self.progressBar.show()

        # 아리마 분석
        history = [x for x in series_studying]
        predictions = list()
        model = sm.tsa.statespace.SARIMAX(history, order=(self.p_value, self.d_value, self.q_value), seasonal_order=(1,1,0,12))
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)
        predictions.append(yhat[0])
        history.append(series_verification.iloc[0])
        self.time = self.progressBar.value()
        self.time += 1
        self.progressBar.setValue(self.time)

        for i in range(1, len_arr):
            model = sm.tsa.statespace.SARIMAX(history, order=(self.p_value, self.d_value, self.q_value), seasonal_order=(1,1,0,12))
            self.model_fit = model.fit()
            yhat = self.model_fit.forecast(steps=1)
            predictions.append(yhat[0])
            history.append(series_verification.iloc[i])
            # print(f'총 {len_arr} 학습 중 {i}번 째 학습 중{yhat}')
            self.rmse_label.setText(f'총 {len_arr} 학습 중 {i}번 째 학습 중')
            self.time += 1
            self.progressBar.setValue(self.time)

        # 예측값
        if len(predictions) == 1:
            self.rmse = '출력 값을 더 많이 설정해주세요'
        else:
            x = pd.Series(predictions, index=series_verification.index)
            self.rmse = sqrt(mean_squared_error(
                series_verification, x))
        self.rmse_label.setText('RMSE : '+str(self.rmse))
        self.arimaresult_label_2.setText('학습량 : '+str(len(series_studying)))

        print('end')
        self.model_status = 'arima'

        # 그래프
        ax = self.fig.add_subplot(1, 1, 1)
        self.fig.set_facecolor("white")
        ax.plot(series_studying, color='blue', label='actual')
        ax.plot(series_verification, color='limegreen', label='actual_verification')
        ax.plot(x, color='red', label='predict')
        ax.set_xlabel(self.comboBox_x.currentText())
        ax.set_ylabel(self.comboBox_y.currentText())
        ax.legend(loc="upper left")
        ax.grid()
        ax2 = self.fig2.add_subplot(1, 1, 1)
        self.fig2.set_facecolor("white")
        ax2.plot(series_verification, color='limegreen', label='actual_verification')
        ax2.plot(x, color='red', label='predict')
        ax2.set_xlabel(self.comboBox_x.currentText())
        ax2.set_ylabel(self.comboBox_y.currentText())
        ax2.legend(loc="upper left")
        ax2.grid()
        plt.show()
        self.canvas.draw()
        self.canvas2.draw()

        # 프로그레스 바
        self.time = 0
        self.progressBar.setValue(self.time)
        self.progressBar.hide()

    def model_save_pushed(self):
        try:
            model = self.model_fit
        except:
            QMessageBox.about(self, "Alert", "먼저 학습을 진행해주세요.")
            return
        savefile = QFileDialog.getSaveFileName(self, '파일저장', '', '(*.pkl)')
        if savefile[0][-7:0] == 'pkl.pkl':
            model.save(savefile[0][:-4])
        else:
            model.save(savefile[0])
        QMessageBox.about(self, "Alert", "저장되었습니다.")
    def LSTM_model_save_pushed(self):
        try:
            savefile = QFileDialog.getSaveFileName(self, '파일저장', '', '(*.h5)')
            print(savefile)
            self.LSTM_model.save(savefile[0][:-3]+'.h5')
        except:
            QMessageBox.about(self, "Alert", "먼저 학습을 진행해주세요.")
            return
        
        QMessageBox.about(self, "Alert", "저장되었습니다.")
    def auto_arima_pushed(self):
        if self.comboBox_x.currentText() == self.comboBox_y.currentText():
            QMessageBox.about(self, "Alert", "x축 y축을 다르게 설정해주세요.")
            return
        elif self.studying_rate.text() == '':
            QMessageBox.about(self, "Alert", "학습 비율을 입력해주세요")
            return
        elif int(self.studying_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "학습 비율을 양수로 입력해주세요")
            return
        elif self.verification_rate.text() == '':
            QMessageBox.about(self, "Alert", "검증 비율을 입력해주세요")
            return
        elif int(self.verification_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "검증 비율을 양수로 입력해주세요")
            return
        elif self.test_rate.text() == '':
            QMessageBox.about(self, "Alert", "테스트 비율을 입력해주세요")
            return
        elif int(self.test_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "테스트 비율을 양수로 입력해주세요")
            return
        self.total_len = int(self.studying_rate.text()) + \
            int(self.verification_rate.text()) + int(self.test_rate.text())
        self.studying_rate_length = len(self.data) * \
            int(self.studying_rate.text()) // self.total_len
        self.verification_rate_length = len(
            self.data) * int(self.verification_rate.text()) // self.total_len
        self.auto_arima_run(pd.Series(list(self.data[self.comboBox_y.currentText(
        )]), index=self.data[self.comboBox_x.currentText()]))

    def auto_arima_run(self, data):
        self.rmse_label.clear()
        self.arimaresult_label_2.clear()
        self.fig.clear()
        self.fig2.clear()
        
        try:
            data.index = pd.to_datetime(
                data.index)
        except:
            pass

        train = data[:self.studying_rate_length + self.verification_rate_length]
        test = data[self.studying_rate_length+self.verification_rate_length:]

        arima_model1 = auto_arima(train, start_p=0, d=1, start_q=0,
                                  max_p=3, max_d=3, max_q=3, m=12,
                                  start_P=0, D=1, start_Q=0,
                                  max_P=3, max_D=3, max_Q=3,
                                  seasonal=True, trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

        self.model_fit = arima_model1

        prediction1 = pd.DataFrame(
            self.model_fit.predict(n_periods=len(test)), index=test.index)
        prediction1.columns = [self.comboBox_y.currentText()]
        rmse1 = sqrt(mean_squared_error(
            test, prediction1[self.comboBox_y.currentText()]))
        self.rmse_label.setText('RMSE : '+str(rmse1))
        self.arimaresult_label_2.setText('학습량 : '+str(len(train)))
        
        ax = self.fig.add_subplot(1, 1, 1)
        self.fig.set_facecolor("white")
        ax.plot(train, color='blue', label='actual')
        ax.plot(test, color='limegreen', label='actual_verfication')
        ax.plot(prediction1, color='red', label='predict')
        ax.set_xlabel(self.comboBox_x.currentText())
        ax.set_ylabel(self.comboBox_y.currentText())
        ax.legend(loc="upper left")
        ax.grid()
        ax2 = self.fig2.add_subplot(1, 1, 1)
        self.fig2.set_facecolor("white")
        ax2.plot(test, color='limegreen', label='actual_verfication')
        ax2.plot(prediction1, color='red', label='predict')
        ax2.set_xlabel(self.comboBox_x.currentText())
        ax2.set_ylabel(self.comboBox_y.currentText())
        ax2.legend(loc="upper left")
        ax2.grid()
        self.canvas.draw()
        self.canvas2.draw()
        self.model_status = 'auto_arima'
        print('end')

    def LSTM_setupUI(self):
        for combo in self.LSTMcomboboxes:
            for idx in range(self.c):
                combo.addItem(self.data.columns[idx])

    def LSTM_Train(self):
        if self.studying_rate.text() == '':
            QMessageBox.about(self, "Alert", "학습 비율을 입력해주세요")
            return
        elif int(self.studying_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "학습 비율을 양수로 입력해주세요")
            return
        elif self.verification_rate.text() == '':
            QMessageBox.about(self, "Alert", "검증 비율을 입력해주세요")
            return
        elif int(self.verification_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "검증 비율을 양수로 입력해주세요")
            return
        elif self.test_rate.text() == '':
            QMessageBox.about(self, "Alert", "테스트 비율을 입력해주세요")
            return
        elif int(self.test_rate.text()) < 0:
            QMessageBox.about(self, "Alert", "테스트 비율을 양수로 입력해주세요")
            return
        elif self.ModelSize_lineText.text() == '':
            QMessageBox.about(self, "Alert", "Model Size 값을 입력해주세요")
            return
        elif self.WindowSize_lineText.text() == '':
            QMessageBox.about(self, "Alert", "Window Size 값을 입력해주세요")
            return
        elif self.PredictSize_lineText.text() == '':
            QMessageBox.about(self, "Alert", "Dense 값을 입력해주세요")
            return
        elif self.Epochs_linetext.text() == '':
            QMessageBox.about(self, "Alert", "Epochs 값을 입력해주세요")
            return
        elif self.BatchSize_lineText.text() == '':
            QMessageBox.about(self, "Alert", "Batch Size 값을 입력해주세요")
            return
        elif self.Patience_lineText.text() == '':
            QMessageBox.about(self, "Alert", "Early Stop 값을 입력해주세요")
            return
    
        stock_file = self.data.loc[:, [self.Date_combobox.currentText()]]
        for combo in self.LSTMcomboboxes:
            if combo == self.Date_combobox:
                continue
            stock_file = pd.merge(
                stock_file, self.data.loc[:, [combo.currentText()]], left_index=True, right_index=True, how='left')

        epochs = int(self.Epochs_linetext.text())
        batch_size = int(self.BatchSize_lineText.text())
        train_ratio = int(self.studying_rate.text()) / (int(self.studying_rate.text()) +
                                                        int(self.verification_rate.text()) + int(self.test_rate.text()))
        valid_ratio = int(self.verification_rate.text()) / (int(self.studying_rate.text()) +
                                                        int(self.verification_rate.text()) + int(self.test_rate.text()))

        window_size = int(self.WindowSize_lineText.text())
        dense = int(self.PredictSize_lineText.text())
        Model_size = int(self.ModelSize_lineText.text())
        patience = int(self.Patience_lineText.text())
        
        LSTM_lossgraph, LSTM_validgraph_origin, LSTM_validgraph_plus, train_part, rmse, self.LSTM_model =  LS.Train(stock_file, train_ratio, valid_ratio, window_size, dense, Model_size, patience, epochs, batch_size)
        # Loss
        self.canvas = FigureCanvas(LSTM_lossgraph)
        # 학습 origin
        self.canvas2 = FigureCanvas(LSTM_validgraph_origin)
        # 학습 확대
        self.canvas3 = FigureCanvas(LSTM_validgraph_plus)
        self.canvas3.hide()

        self.LSTM_Loss_gridLayout.addWidget(self.canvas)
        self.LSTM_Valid_gridLayout.addWidget(self.canvas3)
        self.LSTM_Valid_gridLayout.addWidget(self.canvas2)
        self.Rmse_label.setText(f'RMSE : {rmse}')
        self.Train_label.setText(f'학습량 : {train_part}')

        self.model_status = 'LSTM'
    def LSTM_plus_Btn_pushed(self):
        self.canvas2.hide()
        self.canvas3.show()
    def LSTM_minus_Btn_pushed(self):
        self.canvas3.hide()
        self.canvas2.show()
    def plus_Btn_pushed(self):
        self.canvas.hide()
        self.canvas2.show()

    def minus_Btn_pushed(self):
        self.canvas2.hide()
        self.canvas.show()

    def test_Btn_pushed(self):
        if self.model_status == 'x':
            QMessageBox.about(self, "Alert", "먼저 학습을 시켜주세요.")
            return
        elif self.model_status == 'arima':
            test_result = pd.DataFrame(self.model_fit.forecast(steps=self.len_test), index=self.series_test.index)
            self.fig.clear()
            self.rmse_label.clear()
            self.arimaresult_label_2.clear()
            rmse = sqrt(mean_squared_error(
                self.series_test, test_result))
            self.rmse_label.setText('RMSE : '+str(rmse))
            self.arimaresult_label_2.setText('학습량 : '+str(self.len_test))
            ax = self.fig.add_subplot(1, 1, 1)
            ax.plot(self.series_train, color='blue', label='actual')
            ax.plot(self.series_test, color='limegreen', label='actual_verification')
            ax.plot(test_result, color='red', label='predict')
            ax.set_xlabel(self.comboBox_x.currentText())
            ax.set_ylabel(self.comboBox_y.currentText())
            ax.legend(loc="upper left")
            ax.grid()
            self.fig2.clear()
            ax2 = self.fig2.add_subplot(1, 1, 1)
            ax2.plot(self.series_test, color='limegreen', label='actual_verification')
            ax2.plot(test_result, color='red', label='predict')
            ax2.set_xlabel(self.comboBox_x.currentText())
            ax2.set_ylabel(self.comboBox_y.currentText())
            ax2.legend(loc="upper left")
            ax2.grid()
            self.canvas.draw()
            self.canvas2.draw()
        elif self.model_status == 'auto_arima':
            pass

  