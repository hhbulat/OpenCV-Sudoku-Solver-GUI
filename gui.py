from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QLabel, QPushButton
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QSize
import sys,os
import solver
class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi("ui/trial_4.ui",self)
        self.browse =  self.findChild(QPushButton,"browse")
        self.solve =  self.findChild(QPushButton,"solve")
        self.setStyleSheet("background-color: #a2a4c0 ;")
        self.predict = self.findChild(QLabel,"predict")
        self.solution = self.findChild(QLabel,"solution")
        self.time = self.findChild(QLabel,"time")
        self.scs = self.findChild(QLabel,"scs")
        
        self.browse.clicked.connect(self.browse_files)
        self.solve.clicked.connect(self.solution_solve)

        self.solve.setText('')
        self.solve.setStyleSheet("background-color : #82bb6b; border-radius: 10px; border-color: #82bb6b;")
        self.solve.setIcon(QIcon("ui/icon.png"))
        self.solve.setIconSize(QSize(50,50))

        self.browse.setText('')
        self.browse.setStyleSheet("background-color : #686ebb; border-radius: 10px; border-color: #686ebb; padding-top: 10px; padding-bottom: 15px; padding-left: 5px; padding-right: 5px;")
        self.browse.setIcon(QIcon("ui/search.png"))
        self.browse.setIconSize(QSize(40,40))
        self.show()
    def browse_files(self):
        try:
            fname=QFileDialog.getOpenFileName(self, 'Open file', 'C:')
            empty,_ = solver.gui_predict(fname[0])
            self.pixmap = QPixmap(empty)
            self.predict.setPixmap(self.pixmap)
        except:
            pass
    def solution_solve(self):
        try:
            path,diff,success = solver.gui_solution()
            self.pixmap = QPixmap(path)
            self.solution.setPixmap(self.pixmap)
            os.remove('data/solution.png')
            os.remove('data/predict.png')
        except:
            diff = 0
            success = 'Invalid data.'
        diff=round(diff,3)
        self.time.setText(str(diff) + "seconds")
        self.scs.setText(success)
        if success == 'Solution Succesfully!':
            self.scs.setStyleSheet("background-color : green ; border-radius: 4px ; padding: 2px; color: white; font-size: 15px;")
        else:
            self.scs.setStyleSheet("background-color : red ; border-radius: 4px ; padding: 2px; color: white; font-size: 15px;")
    
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()