# %%
from configparser import Interpolation
import sys
import numpy as np
from rastermap import Rastermap
import pandas as pd
import numpy as np
from scipy import stats
import h5py
import cv2
import tifffile
from scipy import sparse
import tempfile
import os

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

#%%
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, raster, A, C, b, isort):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.tmpdirname = tempfile.TemporaryDirectory()
        print('Created temporary directory', self.tmpdirname)
        self.A = A
        self.C = C
        self.b = b
        self.isort = isort

        play_button = QtWidgets.QPushButton(self._main)
        play_button.setText("Play")
        play_button.clicked.connect(self.play_button_clicked)

        self._main.setWindowTitle("PyQt5 Button Click Example")

        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(10, 10), tight_layout=True))
        layout.addWidget(play_button)
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))


        self._static_ax = static_canvas.figure.subplots()
        self._static_ax.imshow(raster, vmin = 0, vmax=1, aspect='auto', cmap='binary', interpolation="antialiased")


    def play_button_clicked(self):
        t_min, t_max = self._static_ax.get_xlim()
        # y is inverted since we used imshow
        i_max, i_min = self._static_ax.get_ylim()
        print(t_min, t_max, i_min, i_max)
        
        T = C.shape[-1]
        t_min = int(max(0, t_min))
        i_min = int(max(0, i_min))
        t_max = int(min(T, t_max))
        i_max = int(min(T, i_max))

        idx = self.isort[i_min: i_max]

        AC = self.A[:, idx].dot(self.C[idx, t_min: t_max]).T.reshape((-1, 1024, 1024))
        AC += np.repeat(self.b[None, ...], t_max - t_min, axis=0)
        AC *= 255 / AC.max()
        tifffile.imwrite(os.path.join(self.tmpdirname.name, "tmp.tif"), data=AC.astype(np.float32))

        print(AC.shape)

        # for img in AC:
        #     cv2.imshow(img.astype(np.uint8))
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break

# %%
f = h5py.File(r"C:\Users\robert.wong\Downloads\20210921-f1_plane7.hdf5", "r")

A_tmp = f["estimates"]["A"]
C = f["estimates"]["C"][:]
label = f["estimates"]["label"][:]
label = np.array([x.decode("utf-8") for x in label])
print(A_tmp.keys())
A = sparse.csc_matrix((A_tmp["data"], A_tmp["indices"], A_tmp["indptr"]), shape=A_tmp["shape"][:]).copy()
S = f["estimates"]["S_new"][:] > 0
b = f["estimates"]["b0"][:].reshape((1024, 1024))

# S[label != "PV_L"] = 0

f.close()

print(A.shape, C.shape)
AC = (A.dot(C)).T.reshape((-1, 1024, 1024))
print(AC.shape)

# %%
# we run rastermap the same way that the other scikit-learn embedding algorithms work
model = Rastermap(n_components=1, n_X=100).fit(S)
Sfilt = stats.zscore(S[model.isort], axis=1)

# bg = f["estimates"]["b0"][:].reshape((1024, 1024)).T

# %%
qapp = QtWidgets.QApplication(sys.argv)
app = ApplicationWindow(S[model.isort], A, C, b, model.isort)
app.show()
qapp.exec_()

# %%
