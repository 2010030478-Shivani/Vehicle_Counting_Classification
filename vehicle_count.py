



import cv2
import csv
import collections
import numpy as np
from tracker import *


tracker = EuclideanDistTracker()


cap = cv2.VideoCapture('video.mp4')
input_size = 320


confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2


middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))


required_class_index = [2, 3, 5, 7]

detected_classNames = []


modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)



net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')



def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy



temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]



def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    
    center = find_center(x, y, w, h)
    ix, iy = center

    
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    
    cv2.circle(img, center, 2, (0, 0, 255), -1)  
    



def postProcess(outputs, img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        
        outputs = net.forward(outputNames)

        
        postProcess(outputs, img)

        

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    
    
    cap.release()
    cv2.destroyAllWindows()


image_file = 'test.jpg'


def from_static_image(image):
    img = cv2.imread(image)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    
    outputs = net.forward(outputNames)

    
    postProcess(outputs, img)

    
    frequency = collections.Counter(detected_classNames)
    print(frequency)
    
    cv2.putText(img, "Car:        " + str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                font_thickness)
    cv2.putText(img, "Motorbike:  " + str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                font_color, font_thickness)
    cv2.putText(img, "Bus:        " + str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                font_thickness)
    cv2.putText(img, "Truck:      " + str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                font_color, font_thickness)

    cv2.imshow("image", img)

    cv2.waitKey(0)

    
    with open("static-data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
    f1.close()


if __name__ == '__main__':
    
    from_static_image(image_file)





def calculate_accuracy(ground_truth, detected_results, class_labels):
    # Initialize counters for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for ground_truth_entry in ground_truth:
        found_match = False
        for detected_entry in detected_results:
            if ground_truth_entry['class'] == detected_entry['class']:
                # Calculate the intersection over union (IoU) for the two bounding boxes
                intersection = calculate_intersection(ground_truth_entry, detected_entry)
                union = calculate_union(ground_truth_entry, detected_entry)
                iou = intersection / union

                # Define an IoU threshold for a positive match
                iou_threshold = 0.5

                if iou >= iou_threshold:
                    true_positives += 1
                    found_match = True
                    break

        if not found_match:
            false_negatives += 1

    for detected_entry in detected_results:
        if detected_entry['class'] not in [entry['class'] for entry in ground_truth]:
            false_positives += 1

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def calculate_intersection(entry1, entry2):
    x1 = max(entry1['x'], entry2['x'])
    y1 = max(entry1['y'], entry2['y'])
    x2 = min(entry1['x'] + entry1['width'], entry2['x'] + entry2['width'])
    y2 = min(entry1['y'] + entry1['height'], entry2['y'] + entry2['height'])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height

def calculate_union(entry1, entry2):
    area1 = entry1['width'] * entry1['height']
    area2 = entry2['width'] * entry2['height']
    return area1 + area2 - calculate_intersection(entry1, entry2)

# Example ground truth and detected results
ground_truth = [
    {'x': 50, 'y': 60, 'width': 70, 'height': 40, 'class': 'car'},
    {'x': 100, 'y': 80, 'width': 60, 'height': 30, 'class': 'motorbike'}
]

detected_results = [
    {'x': 50, 'y': 60, 'width': 70, 'height': 40, 'class': 'car'},
    {'x': 110, 'y': 90, 'width': 55, 'height': 25, 'class': 'motorbike'}
]

class_labels = ['car', 'motorbike']

precision, recall, f1_score = calculate_accuracy(ground_truth, detected_results, class_labels)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
