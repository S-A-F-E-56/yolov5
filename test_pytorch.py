import sys
from pathlib import Path
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh, increment_path
from utils.plots import save_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Tambahkan YOLOv5 ke sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Ganti [1] dengan level yang sesuai jika script berada dalam subdirektori
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Tangkap gambar dari kamera laptop
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite('capture.jpg', frame)
    print("Image captured and saved as capture.jpg")
else:
    print("Failed to capture image")
    cap.release()
    sys.exit()

cap.release()

# Set opsi untuk deteksi
class Opt:
    weights = 'best.pt'
    source = 'capture.jpg'
    img_size = 640
    conf_thres = 0.5
    iou_thres = 0.45
    device = ''
    view_img = True
    save_txt = True
    save_conf = True
    project = '.'
    name = 'results'
    exist_ok = True
    dnn = False
    half = False
    data = None

opt = Opt()

# Inisialisasi dan jalankan deteksi
device = select_device(opt.device)
model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)

# Load image
img = cv2.imread(opt.source)

# Run inference
img_tensor = torch.from_numpy(img).to(device)
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)

# Forward pass
pred = model(img_tensor, augment=False, visualize=False)

# Apply NMS (Non-Maximum Suppression)
pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)

# Process detections
for i, det in enumerate(pred):
    if len(det):
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{model.names[int(cls)]} {conf:.2f}'
            save_one_box(xyxy, img, label=label, color=(255, 0, 0), line_thickness=2)

# Save results
save_path = str(Path(opt.project) / opt.name / 'result.jpg')
cv2.imwrite(save_path, img)

# Display results
cv2.imshow('Detection Results', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
