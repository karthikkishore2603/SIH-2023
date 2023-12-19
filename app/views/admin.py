from .. import app, views
from ..detector import WeaponDetector
from time import time
from flask import (
    render_template,
    request,
    redirect,
    url_for,
    make_response,
    json,
    Response,
    jsonify,Flask
)
#from camera import VideoCamera
import cv2, numpy as np
@app.get("/")
def index():
    return render_template("login.html")


@app.post("/")
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username == "kk" and password == "kk":
        print("Login Successful")
        return redirect(url_for("dashboard"))
    else:
        print("Login Failed")
        return redirect(url_for("index"))
    

@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.get("/camera_view")
def camera_view():
    return render_template("cam.html")





def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n") 

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.detector = WeaponDetector(capture_index=0, model_name='app/views/gnf_best.pt')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (416, 416))
        start_time = time()
        results = self.detector.score_frame(frame)
        frame = self.detector.plot_boxes(results, frame)

        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    
@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoCamera()), mimetype="multipart/x-mixed-replace; boundary=frame")


# import torch
# import numpy as np
# import cv2
# from time import time

# class KnifeDetection:
#     def __init__(self, capture_index, model_name):
#         self.capture_index = capture_index
#         self.model = self.load_model(model_name)
#         self.classes = self.model.names
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("Using Device: ", self.device)

#     def get_video_capture(self):
#         return cv2.VideoCapture(self.capture_index)
    
#     def load_model(self, model_name):
#         if model_name:
#             model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
#         else:
#             model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#         return model
    
#     def score_frame(self, frame):
#         self.model.to(self.device)
#         frame = [frame]
#         results = self.model(frame)
#         results = self.model(frame)
#         labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#         return labels, cord
    
#     def class_to_label(self, x):
#         return self.classes[int(x)]
    
#     def plot_boxes(self, results, frame):
#         labels, cord = results
#         n = len(labels)
#         x_shape, y_shape = frame.shape[1], frame.shape[0]
#         for i in range(n):
#             row = cord[i]
#             if row[4] >= 0.2:
#                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#                 bgr = (0, 255, 0)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                 cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

#         return frame
    
#     def __call__(self):
#         cap = self.get_video_capture()
#         assert cap.isOpened()

#         while True:

#             ret, frame = cap.read()
#             assert ret

#             frame = cv2.resize(frame, (416, 416))

#             start_time = time()
#             results = self.score_frame(frame)
#             frame = self.plot_boxes(results, frame)

#             end_time = time()
#             fps = 1/np.round(end_time - start_time, 2)

#             cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#             cv2.imshow('YOLOv5 Detection', frame)

#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break

#         cap.release()

# #Create a new object and execute.
# detector = KnifeDetection(capture_index=0, model_name='gnf_best.pt')
# detector()


# from flask import Flask, render_template, Response
# import cv2
# from time import time
# import numpy as np
# import torch

# app = Flask(__name__)

# class KnifeDetection:
#     def __init__(self, capture_index, model_name):
#         self.capture_index = capture_index
#         self.model = self.load_model(model_name)
#         self.classes = self.model.names
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("Using Device: ", self.device)


#     def get_video_capture(self):
#         return cv2.VideoCapture(self.capture_index)
    
#     def load_model(self, model_name):
#         if model_name:
#             model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
#         else:
#             model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#         return model
    
#     def score_frame(self, frame):
#         self.model.to(self.device)
#         frame = [frame]
#         results = self.model(frame)
#         results = self.model(frame)
#         labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#         return labels, cord
    
#     def class_to_label(self, x):
#         return self.classes[int(x)]
    
#     def plot_boxes(self, results, frame):
#         labels, cord = results
#         n = len(labels)
#         x_shape, y_shape = frame.shape[1], frame.shape[0]
#         for i in range(n):
#             row = cord[i]
#             if row[4] >= 0.2:
#                 x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
#                 bgr = (0, 255, 0)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                 cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

#         return frame
    
#     def __call__(self):
#         cap = self.get_video_capture()
#         assert cap.isOpened()

#         while True:

#             ret, frame = cap.read()
#             assert ret

#             frame = cv2.resize(frame, (416, 416))

#             start_time = time()
#             results = self.score_frame(frame)
#             frame = self.plot_boxes(results, frame)

#             end_time = time()
#             fps = 1/np.round(end_time - start_time, 2)

#             cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#             cv2.imshow('YOLOv5 Detection', frame)

#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break

#         cap.release()

#     def generate_frames(self):
#         cap = self.get_video_capture()
#         assert cap.isOpened()

#         while True:
#             ret, frame = cap.read()
#             assert ret

#             frame = cv2.resize(frame, (416, 416))

#             start_time = time()
#             results = self.score_frame(frame)
#             frame = self.plot_boxes(results, frame)

#             end_time = time()
#             fps = 1 / np.round(end_time - start_time, 2)

#             cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#             ret, jpeg = cv2.imencode('.jpg', frame)
#             frame = jpeg.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     @app.route('/')
#     def index(self):
#         return render_template('cam.html')

#     @app.route('/video_feed')
#     def video_feed(self):
#         return Response(self.generate_frames(),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')

# # Create a new object and execute.
# detector = KnifeDetection(capture_index=0, model_name="C:/Users/DELL/OneDrive/Documents/GitHub/SIH-2023/app/views/gnf_best.pt")
# detector()

# if __name__ == '__main__':
#     app.run(debug=True)
