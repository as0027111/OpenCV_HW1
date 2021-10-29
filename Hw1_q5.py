from PyQt5 import QtWidgets, uic
import sys
from lib import Question5

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("Hw1_q5.ui", self)
        
        self.q5 = Question5.Q5()
        # Find the button with the name ""
        self.p1_Btn = self.findChild(QtWidgets.QPushButton, 'problem1')
        self.p1_Btn.clicked.connect(self.q5.Q5_1)
        self.p2_Btn = self.findChild(QtWidgets.QPushButton, 'problem2')
        self.p2_Btn.clicked.connect(self.q5.Q5_2)
        self.p3_Btn = self.findChild(QtWidgets.QPushButton, 'problem3')
        self.p3_Btn.clicked.connect(self.q5.Q5_3)
        self.p4_Btn = self.findChild(QtWidgets.QPushButton, 'problem4')
        self.p4_Btn.clicked.connect(self.q5.Q5_4)
        
        self.problem5.clicked.connect(self.pass_to_Q5)

        self.text = self.findChild(QtWidgets.QLineEdit, "select_Num")
        self.text.setText("1")

        self.show()

    def pass_to_Q5(self, MainWindow):
            # This is executed when the button is pressed
            selected_target_num = int(self.text.text())
            self.q5.Q5_5(selected_target_num)
            # print('printButtonPressed')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()