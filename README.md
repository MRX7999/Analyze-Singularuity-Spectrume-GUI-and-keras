# Singularity
 Analyze Singularuity Spectrume with Keras and GUI
# Singular Spectrum Analysis

This is a Python application that performs Singular Spectrum Analysis (SSA) on a given time series data and displays the results in a graphical user interface (GUI). The application uses a neural network model to predict the future values of the time series data and displays the predicted values in real-time on a graph.

## Requirements

- Python 3.8 or higher
- numpy
- matplotlib
- PyQt5
- tensorflow
- scikit-learn

## Installation

1. Clone the repository to your local machine.
2. Install the required packages by running `pip install -r requirements.txt` in the terminal.
3. Run the application by running `python main.py` in the terminal.

## Usage

1. Open the application by running `python main.py` in the terminal.
2. Click on the "Open File" button to select a CSV file containing time series data.
3. The application will display the raw data on a graph.
4. The user can select the type of scaler to use for normalization of the data.
5. The user can select the number of epochs to use for training the neural network model.
6. The user can add additional plots by clicking on the "Add Plot" button.
7. The user can export the current plot to a PNG file by clicking on the "Export Plot" button.
8. The user can save the predicted values to a CSV file by clicking on the "Save Results" button.
9. The user can save the neural network model to a file by clicking on the "Save Model" button.
10. The user can load a previously saved neural network model by clicking on the "Load Model" button.

## License

This application is licensed under the MIT License. See the LICENSE file for more information.