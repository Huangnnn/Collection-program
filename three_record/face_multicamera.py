import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QObject, pyqtSignal
from face_multicamera_mainwindow import Ui_MainWindow

import os
import time
import datetime as dt
import numpy as np
# import matplotlib.pyplot as plt
import threading as th
import ctypes
import inspect

import re

# First import the library
import pyrealsense2 as rs
import cv2
# from skimage import io
# from PIL import Image

DELAY = 0



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the user interface from Designer.
        self.setupUi(self)

        self.pushButton_record_start.setDisabled(True)
        self.pushButton_record_stop.setDisabled(True)

        self.dis_update.connect(self.camera_view)
        self.pushButton_record_start.clicked.connect(self.pushButton_record_start_clicked)
        self.pushButton_record_stop.clicked.connect(self.pushButton_record_stop_clicked)


        self.thread_camera = None
        self.takePhotos = False
        self.preview = True
        self.user_id = ''

        self.normal_start = False

        # self.setWindowIcon(QIcon('logo_face.png'))
        self.move(0, 0)

    # 在对应的页面类的内部，与def定义的函数同级
    dis_update = pyqtSignal(QPixmap)

    def pushButton_record_start_clicked(self):

        self.user_id = self.lineEdit_id.text()
        if self.user_id == '':
            self.user_id = dt.datetime.now().strftime('%Y_%m%d_%H%M%S')
        print('self.user_id:', self.user_id)

        self.pushButton_record_start.setEnabled(False)
        self.pushButton_record_stop.setEnabled(True)
        self.takePhotos = True


    def pushButton_record_stop_clicked(self):
        self.pushButton_record_start.setEnabled(True)
        self.pushButton_record_stop.setEnabled(False)
        self.takePhotos = False
        # 清空文本输入框
        self.lineEdit_id.setText('')


    # 添加一个退出的提示事件
    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮：Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
              第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if reply == QMessageBox.Yes:
            if self.normal_start:
                self.stop_thread(self.thread_camera)
            event.accept()
        else:
            event.ignore()


    def open_camera(self):
        # target选择开启摄像头的函数
        self.thread_camera = th.Thread(target=self.open_realsense)
        self.thread_camera.start()
        print('Open Camera')

    def camera_view(self, c):
        # 调用setPixmap函数设置显示Pixmap
        self.label_show.setPixmap(c)
        # 调用setScaledContents将图像比例化显示在QLabel上
        self.label_show.setScaledContents(True)

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    # def drawCross(img, point=(640,360), color=(0,0,255), size=64, thickness=3):
    def drawCross(self, img, point=(320, 180), color=(0, 0, 255), size=32, thickness=3):
        # 绘制横线
        cv2.line(img, (point[0] - round(size / 2), point[1]), (point[0] + round(size / 2), point[1]), color, thickness,
                 8,
                 0)
        # 绘制竖线
        cv2.line(img, (point[0], point[1] - round(size / 2)), (point[0], point[1] + round(size / 2)), color, thickness,
                 8,
                 0)
        return

    def open_realsense(self):
        print('open_realsense')
        rs.context.devices

        # Create a pipeline
        pipeline1 = rs.pipeline()
        pipeline2 = rs.pipeline()
        pipeline3 = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config1 = rs.config()
        config2 = rs.config()
        config3 = rs.config()

        serials = ['025522062091', '025522060999', '039422061177']
        serials = ['127122062881', '204222065965', '025522062091']

        config1.enable_device(serials[0])
        config2.enable_device(serials[1])
        config3.enable_device(serials[2])

        config1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config3.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config3.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        profile1 = pipeline1.start(config1)
        profile2 = pipeline2.start(config2)
        profile3 = pipeline3.start(config3)



        sensor1 = pipeline1.get_active_profile().get_device().query_sensors()[1]
        sensor2 = pipeline2.get_active_profile().get_device().query_sensors()[1]
        sensor3 = pipeline3.get_active_profile().get_device().query_sensors()[1]

        # 关闭白平衡
        sensor1.set_option(rs.option.enable_auto_white_balance, False)
        sensor2.set_option(rs.option.enable_auto_white_balance, False)
        sensor3.set_option(rs.option.enable_auto_white_balance, False)

        profile1_depth = profile1.get_stream(rs.stream.depth)
        camera0_intr1 = profile1_depth.as_video_stream_profile().get_intrinsics()
        print('camera 0 depth intrinsics:', camera0_intr1)
        profile1_color = profile1.get_stream(rs.stream.color)
        camera0_intr2 = profile1_color.as_video_stream_profile().get_intrinsics()
        print('camera 0 color intrinsics:', camera0_intr2)

        profile2_depth = profile2.get_stream(rs.stream.depth)
        camera1_intr1 = profile2_depth.as_video_stream_profile().get_intrinsics()
        print('camera 1 depth intrinsics:', camera1_intr1)
        profile2_color = profile2.get_stream(rs.stream.color)
        camera1_intr2 = profile2_color.as_video_stream_profile().get_intrinsics()
        print('camera 1 color intrinsics:', camera1_intr2)

        profile3_depth = profile3.get_stream(rs.stream.depth)
        camera2_intr1 = profile3_depth.as_video_stream_profile().get_intrinsics()
        print('camera 2 depth intrinsics:', camera2_intr1)
        profile3_color = profile3.get_stream(rs.stream.color)
        camera2_intr2 = profile3_color.as_video_stream_profile().get_intrinsics()
        print('camera 2 color intrinsics:', camera2_intr2)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile1.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # config filters
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 1)

        hdr_merge = rs.hdr_merge()

        threshold_filter = rs.threshold_filter()
        threshold_filter.set_option(rs.option.min_distance, 0.1)
        threshold_filter.set_option(rs.option.max_distance, 4.0)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 0)

        temporal = rs.temporal_filter()

        hole_filling = rs.hole_filling_filter()

        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align1 = rs.align(align_to)
        align2 = rs.align(align_to)
        align3 = rs.align(align_to)

        depth_image1 = None
        depth_image2 = None
        depth_image3 = None
        color_image1 = None
        color_image2 = None
        color_image3 = None

        # Create colorizer object
        colorizer = rs.colorizer()

        start_time = time.time()
        time_point_past = start_time
        frame_number = 0

        # Streaming loop
        try:
            while True:

                # Get frameset of color and depth
                frames1 = pipeline1.wait_for_frames()
                frames2 = pipeline2.wait_for_frames()
                frames3 = pipeline3.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames1 = align1.process(frames1)
                aligned_frames2 = align2.process(frames2)
                aligned_frames3 = align3.process(frames3)


                # Get aligned frames
                aligned_depth_frame1 = aligned_frames1.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame1 = aligned_frames1.get_color_frame()
                aligned_depth_frame2 = aligned_frames2.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame2 = aligned_frames2.get_color_frame()
                aligned_depth_frame3 = aligned_frames3.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame3 = aligned_frames3.get_color_frame()


                frame = decimation.process(aligned_depth_frame1)
                frame = hdr_merge.process(frame)
                frame = threshold_filter.process(frame)
                frame = depth_to_disparity.process(frame)
                frame = spatial.process(frame)
                frame = temporal.process(frame)
                # frame = hole_filling.process(frame)
                aligned_depth_frame1 = disparity_to_depth.process(frame)

                frame = decimation.process(aligned_depth_frame2)
                frame = hdr_merge.process(frame)
                frame = threshold_filter.process(frame)
                frame = depth_to_disparity.process(frame)
                frame = spatial.process(frame)
                frame = temporal.process(frame)
                # frame = hole_filling.process(frame)
                aligned_depth_frame2 = disparity_to_depth.process(frame)

                frame = decimation.process(aligned_depth_frame3)
                frame = hdr_merge.process(frame)
                frame = threshold_filter.process(frame)
                frame = depth_to_disparity.process(frame)
                frame = spatial.process(frame)
                frame = temporal.process(frame)
                # frame = hole_filling.process(frame)
                aligned_depth_frame3 = disparity_to_depth.process(frame)

                # Validate that both frames are valid
                if not aligned_depth_frame1 or not color_frame1:
                    continue

                # Render images
                depth_image1 = np.asanyarray(colorizer.colorize(aligned_depth_frame1).get_data())
                color_image1 = np.asanyarray(color_frame1.get_data())
                depth_image2 = np.asanyarray(colorizer.colorize(aligned_depth_frame2).get_data())
                color_image2 = np.asanyarray(color_frame2.get_data())
                depth_image3 = np.asanyarray(colorizer.colorize(aligned_depth_frame3).get_data())
                color_image3 = np.asanyarray(color_frame3.get_data())

                depth_image1_resized = cv2.resize(depth_image1, (640,360))
                depth_image2_resized = cv2.resize(depth_image2, (640,360))
                depth_image3_resized = cv2.resize(depth_image3, (640,360))


                color_image1_resized = cv2.resize(color_image1, (640,360))
                color_image2_resized = cv2.resize(color_image2, (640,360))
                color_image3_resized = cv2.resize(color_image3, (640,360))

                # 为显示的中间彩色图片加矩形框和图片中心十字标识
                self.drawCross(color_image2_resized)
                cv2.rectangle(color_image2_resized, (240, 80), (400, 280), (0, 255, 0))


                # images1 = np.hstack((color_image1[0:720, 320:960], color_image2[0:720, 320:960],color_image3[0:720, 320:960]))
                images1 = np.hstack((color_image1_resized, color_image2_resized, color_image3_resized))
                images2 = np.hstack((depth_image1_resized, depth_image2_resized, depth_image3_resized))
                images = np.vstack((images1,images2))

                qimage = QImage(images, 1920, 720, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qimage)
                self.dis_update.emit(pixmap)

                if (self.takePhotos == True):

                    now_date = dt.datetime.now().strftime('%F')
                    now_time = dt.datetime.now().strftime('%F_%H%M%S_%f')

                    path_ok = os.path.exists(now_date)
                    if (path_ok == False):
                        os.mkdir(now_date)

                    if (os.path.isdir(now_date)):

                        depth_full_path1 = ''
                        color_full_path1 = ''
                        depth_full_path2 = ''
                        color_full_path2 = ''
                        depth_full_path3 = ''
                        color_full_path3 = ''

                        # if (re.match('^[a-zA-Z0-9_]*$', id) and (id != '')):
                        if re.match('^[\u4E00-\u9FA5a-zA-Z0-9_]*$', self.user_id):

                            dir_str = os.path.join('./', now_date, self.user_id)
                            path_ok = os.path.exists(dir_str)
                            if (path_ok == False):
                                os.mkdir(dir_str)
                                # print('path ok False')
                            depth_full_path1 = os.path.join(dir_str, now_time + '_depth_0.png')
                            color_full_path1 = os.path.join(dir_str, now_time + '_color_0.png')
                            depth_full_path2 = os.path.join(dir_str, now_time + '_depth_1.png')
                            color_full_path2 = os.path.join(dir_str, now_time + '_color_1.png')
                            depth_full_path3 = os.path.join(dir_str, now_time + '_depth_2.png')
                            color_full_path3 = os.path.join(dir_str, now_time + '_color_2.png')

                            cv2.imencode('.png', np.asanyarray(aligned_depth_frame1.get_data()))[1].tofile(depth_full_path1)
                            cv2.imencode('.png', np.asanyarray(aligned_depth_frame2.get_data()))[1].tofile(depth_full_path2)
                            cv2.imencode('.png', np.asanyarray(aligned_depth_frame3.get_data()))[1].tofile(depth_full_path3)
                            cv2.imencode('.png', color_image1)[1].tofile(color_full_path1)
                            cv2.imencode('.png', color_image2)[1].tofile(color_full_path2)
                            cv2.imencode('.png', color_image3)[1].tofile(color_full_path3)


                elapsed_time = (time.time() - start_time)
                frame_number = frame_number + 1
                frame_rate_avg = frame_number / elapsed_time
                frame_rate_rt = 1 / (elapsed_time - time_point_past)
                time_point_past = elapsed_time  # 把上个周期的时间节点保存下来
                self.label_time.setText('%.2f' % elapsed_time)
                self.label_frame.setText(str(frame_number))
                self.label_frame_rate_avg.setText('%.2f' % frame_rate_avg)
                self.label_frame_rate_rt.setText('%.2f' % frame_rate_rt)
                # print(elapsed_time, frame_number, frame_rate)


                time.sleep(DELAY)
        finally:
            pipeline1.stop()
            pipeline2.stop()
            pipeline3.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()

    ctx = rs.context()
    devs = ctx.query_devices()
    if devs.size() == 3:
        w.pushButton_record_start.setEnabled(True)
        w.open_camera()
        w.normal_start = True
    else:
        str1 = '相机0序列号：025522062091, 相机1序列号：025522060999, 相机2序列号：039422061177\r\n'
        str2 = '当前已连接相机：'
        if len(ctx.devices) > 0:
            for d in ctx.devices:
                str2 = str2 + d.get_info(rs.camera_info.serial_number)

        w.label_show.setText(str1+str2)
        w.normal_start = False

    # thread_camera = th.Thread(target=w.open_realsense)
    # thread_camera.start()

    print('face_multicamera started!')

    sys.exit(app.exec_())
