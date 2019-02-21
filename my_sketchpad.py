import sys
#import recognition
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt, QPoint


class Winform(QWidget):
    def __init__(self, parent=None):
        super(Winform, self).__init__(parent)
        #设置标题
        self.setWindowTitle("绘图例子")
        #实例化QPixmap类
        self.pix = QPixmap()
        #起点，终点
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        #初始化
        self.initUi()

    def initUi(self):
        # 窗口大小设置为600*500
        self.resize(600, 500)

        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(600, 400)
        self.pix.fill(Qt.white)

    def paintEvent(self, event):
        self.pp = QPainter(self.pix)
        # 根据鼠标指针前后两个位置绘制直线
        self.pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        #绘制画布到窗口指定位置处
        painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Winform()
    form.show()
    sys.exit(app.exec_())