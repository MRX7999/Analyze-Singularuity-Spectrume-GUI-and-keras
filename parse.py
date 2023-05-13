
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # creating a central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
       

       # initializing the scaler
        self.scaler = MinMaxScaler() # Added initialization of scaler attribute
        # creating a layout for central widget
        layout = QtWidgets.QVBoxLayout(central_widget)

        # creating a figure and canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.canvas = FigureCanvas(self.figure)

        # adding elements to the layout
        layout.addWidget(self.canvas)

        # setting up the menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_file_action = file_menu.addAction('Open File')
        open_file_action.triggered.connect(self.open_file)
        save_file_action = file_menu.addAction('Save Results')
        save_file_action.triggered.connect(self.save_results)
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        # creating and setting up a toolbar
        tool_bar = QtWidgets.QToolBar()
        self.addToolBar(tool_bar)

        # creating and setting up a status bar
        status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(status_bar)

        # initializing the figure
        self.initialize_figure()

        # creating a timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(1000)

        self.setWindowTitle("Singular Spectrum Analysis")
        self.show()

        # creating the neural model
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=[1]),
            keras.layers.Dense(1)
        ])
        self.model = model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.epochs = 1 # Default epochs value

        # creating a QSpinBox for epochs selection
        self.epochs_spinbox = QtWidgets.QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(self.epochs)
        self.epochs_spinbox.valueChanged.connect(self.set_epochs)
        layout.addWidget(self.epochs_spinbox)

        # creating a QComboBox for epoch selection
        self.epoch_combo = QtWidgets.QComboBox()
        self.epoch_combo.addItems(['1', '10', '50', '100', '500', '1000'])
        self.epoch_combo.setCurrentIndex(0)
        self.epoch_combo.currentIndexChanged.connect(self.set_epochs)
        layout.addWidget(self.epoch_combo)

        # creating a QComboBox for scaler selection
        self.scaler_combo = QtWidgets.QComboBox()
        self.scaler_combo.addItems(['MinMaxScaler', 'StandardScaler'])
        self.scaler_combo.setCurrentIndex(0)
        self.scaler_combo.currentIndexChanged.connect(self.set_scaler)
        layout.addWidget(self.scaler_combo)

        # creating a QCheckBox for model saving
       # self.save_model_checkbox = QtWidgets.QCheckBox('Save Model')
       # self.save_model_checkbox.setChecked(False)
       # layout.addWidget(self.save_model_checkbox)

        # creating a QPushButton for model saving
        self.save_model_button = QtWidgets.QPushButton('Save Model')
        #self.save_model_button.setEnabled(False)
        self.save_model_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_model_button)

        # creating a QPushButton for model loading
        self.load_model_button = QtWidgets.QPushButton('Load Model')
        self.load_model_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_button)

        # creating a QTabWidget for multiple plots
        self.plot_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.plot_tabs)

        # creating a QPushButton for plot adding
        self.add_plot_button = QtWidgets.QPushButton('Add Plot')
        self.add_plot_button.clicked.connect(self.add_plot)
        layout.addWidget(self.add_plot_button)

        # creating a QSpinBox for plot size selection
        self.plot_size_spinbox = QtWidgets.QSpinBox()
        self.plot_size_spinbox.setMinimum(1)
        self.plot_size_spinbox.setMaximum(100)
        self.plot_size_spinbox.setValue(8)
        self.plot_size_spinbox.valueChanged.connect(self.set_plot_size)
        layout.addWidget(self.plot_size_spinbox)

        # creating a QPushButton for plot exporting
        self.export_plot_button = QtWidgets.QPushButton('Export Plot')
        self.export_plot_button.clicked.connect(self.export_plot)
        layout.addWidget(self.export_plot_button)
        
        

    def initialize_figure(self):
        """
        Инициализация графика.
        """
        # creating some dummy data
        self.data = np.random.randn(100)
        self.time = np.arange(100)

        # plotting the data
        self.ax1.plot(self.time, self.data, color='k')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Data')
        self.ax1.set_title('Raw Data')
        self.ax2.plot([], [], color='r')
        self.ax2.plot(self.time, self.data, color='k')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Data')
        self.ax2.set_title('Reconstructed Data')

        # scaling the data
        self.scaled_data = self.scaler.fit_transform(self.data.reshape(-1, 1))
    
    def set_scaler(self, index):
        """
        Обработка изменения типа нормализации данных.
        """
        if index == 0:
            self.scaler = MinMaxScaler()
        elif index == 1:
            self.scaler = StandardScaler()
        # re-scaling the data with the new scaler
        self.scaled_data = self.scaler.fit_transform(self.data.reshape(-1, 1))

    def save_model(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Model", "", "H5 Files (*.h5)", options=options)
        if file_name:
            self.model.save(file_name)

    def update_figure(self):
        """
        Обновление графика с помощью нейронной сети.
        """
        # updating the data using the neural model
        input_data = self.scaled_data[:-1].reshape(-1, 1)
        target_data = self.scaled_data[1:].reshape(-1, 1)
        self.model.fit(input_data, target_data, epochs=self.epochs, verbose=0) # Added epochs parameter
        predicted_data = self.model.predict(input_data).flatten()

        # inverse scaling the predicted data
        predicted_data = self.scaler.inverse_transform(predicted_data.reshape(-1, 1)).flatten()

        # updating the plot
        self.ax1.lines[0].set_data(self.time[:-1], predicted_data)
        self.ax2.lines[0].set_data(self.time[:-1], predicted_data)
        self.ax2.lines[1].set_data(np.concatenate([self.time[:-1], [self.time[-1]]]), np.concatenate([predicted_data, [predicted_data[-1]]]))
        self.ax2.relim()
        self.ax2.autoscale_view(True, True, True)
        self.canvas.draw()

    def open_file(self):
        """
        Открытие файла с данными для отображения на графике.
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv)')
        if file_path:
            data = np.loadtxt(file_path, delimiter=',')
            self.data = data[:, 1]
            self.time = data[:, 0]
            self.initialize_figure()

    def save_results(self):
        """
        Сохранение результатов в файл.
        """
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Results','', 'CSV Files (*.csv)')
        if file_path:
            # saving the original and predicted data to a CSV file
            original_data = np.concatenate([self.time.reshape(-1, 1), self.data.reshape(-1, 1)], axis=1)
            predicted_data = np.concatenate([self.time[:-1].reshape(-1, 1), self.ax1.lines[0].get_ydata()[:-1].reshape(-1, 1)], axis=1)
            np.savetxt(file_path, np.concatenate([original_data, predicted_data], axis=1), delimiter=',')

            # saving the model if the checkbox is checked
            if self.save_model_checkbox.isChecked():
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model', '', 'H5 Files (*.h5)')
                if file_path:
                    self.model.save(file_path)
    
    def save_model(self):
        """
        Сохранение модели в файл.
        """
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model', '', 'H5 Files (*.h5)')
        if file_path:
            self.model.save(file_path)

    def load_model(self):
        """
        Загрузка модели из файла.
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Model', '', 'H5 Files (*.h5)')
        if file_path:
            self.model = keras.models.load_model(file_path)
            self.save_model_button.setEnabled(True)

    def set_epochs(self, value):
        """
        Обработка изменения количества эпох обучения модели.
        """
        self.epochs = value
    
    def add_plot(self):
        """
        Добавление нового графика в QTabWidget.
        """
        # creating a new figure and canvas
        new_figure = plt.figure(figsize=(8, 6))
        new_canvas = FigureCanvas(new_figure)

        # adding the canvas to a new tab
        new_tab = QtWidgets.QWidget()
        new_tab_layout = QtWidgets.QVBoxLayout(new_tab)
        new_tab_layout.addWidget(new_canvas)
        self.plot_tabs.addTab(new_tab, f'Plot {self.plot_tabs.count()+1}')

        # plotting some dummy data
        new_data = np.random.randn(100)
        new_time = np.arange(100)
        new_ax = new_figure.add_subplot(111)
        new_ax.plot(new_time, new_data, color='k')
        new_ax.set_xlabel('Time')
        new_ax.set_ylabel('Data')
        new_ax.set_title(f'Plot {self.plot_tabs.count()}')

        # scaling the data
        new_scaled_data = self.scaler.fit_transform(new_data.reshape(-1, 1))

        # creating a timer for the new plot
        new_timer = QtCore.QTimer()
        new_timer.timeout.connect(lambda: self.update_new_plot(new_ax, new_scaled_data))
        new_timer.start(1000)

    def update_new_plot(self, ax, scaled_data):
        """
        Обновление графика в новой вкладке.
        """
        # updating the data using the neural model
        input_data = scaled_data[:-1].reshape(-1, 1)
        target_data = scaled_data[1:].reshape(-1, 1)
        self.model.fit(input_data, target_data, epochs=self.epochs, verbose=0)
        predicted_data = self.model.predict(input_data).flatten()

        # inverse scaling the predicted data
        predicted_data = self.scaler.inverse_transform(predicted_data.reshape(-1, 1)).flatten()

        # updating the plot
        ax.lines[0].set_data(np.arange(len(predicted_data)), predicted_data)
        ax.relim()
        ax.autoscale_view(True, True, True)

    def set_plot_size(self, value):
        """
        Изменение размера графиков.
        """
        self.figure.set_size_inches(value, value/1.5)

    def export_plot(self):
        """
        Экспорт графика в файл.
        """
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Plot', '', 'PNG Files (*.png)')
        if file_path:
            self.figure.savefig(file_path)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    app.exec_()
