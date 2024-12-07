from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)


results = model.train(data=r"C:\Users\sarfa\Documents\PythonProject\image"
                           r"-classification-yolov8\data\weather_dataset",
                      epochs=20, imgsz=64)
