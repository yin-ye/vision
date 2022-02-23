import cv2
import numpy as np

def resizeImage(img, scale=20):
    width = int(img.shape[1] * scale/100)
    height = int(img.shape[0] * scale / 100)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img

def getBboxCoordinates(bboxes, index, img):
    # extract width and height of image to adapt the
    # bounding box coordinates with image details
    height, width, _ = img.shape
    bbox = bboxes[0, 0, index]
    x1 = int(bbox[3] * width)
    y1 = int(bbox[4] * height)
    x2 = int(bbox[5] * width)
    y2 = int(bbox[6] * height)
    return x1, y1, x2, y2

def extractMask(bboxes, index, masks, roi):
    bbox = bboxes[0, 0, index]
    # the class id tells the type of object detected
    class_id = bbox[1]
    mask = masks[index, int(class_id)]
    # get the region of interest width and height to adjust mask size
    roi_height, roi_width, _ = roi.shape
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

    return mask, class_id

def getDetectionScore(bboxes, index):
    bbox = bboxes[0, 0, index]
    detection_score = bbox[2]
    return detection_score

def createRandomColourArray(num_detected_objects):
    num_of_channels = 3
    colour_array = np.random.randint(0, 255, (num_detected_objects, num_of_channels))
    return colour_array

if __name__ == "__main__":
    img = cv2.imread('images/pexels-jean-van-der-meulen-1559388.jpg')

    img = resizeImage(img, 90)

    # Load the pretrained Mask RCNN model
    network = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                        "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    # input the derived blob images into the mask rcnn network
    # and output the bounding boxes and masks from the last layer
    blob_image = cv2.dnn.blobFromImage(img, swapRB=True)
    network.setInput(blob_image)
    bboxes, masks = network.forward(["detection_out_final", "detection_masks"])
    num_detected_objects = bboxes.shape[2]

    # create blank image to draw mask on
    h, w, _ = img.shape
    empty_img = np.zeros((h, w, 3), np.uint8)

    # loop through the number of each detected object in the image and
    # output the masks and bounding box
    for i in range(num_detected_objects):
        detection_score = getDetectionScore(bboxes, i)
        if detection_score < 0.6:
            continue
        
        # extract bounding box coordinates from individual boxes and draw over the image
        x1, y1, x2, y2 = getBboxCoordinates(bboxes, i, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # get the region of interest from the empty (black) image
        roi = empty_img[y1: y2, x1: x2]

        # find the contours of the mask assign a random colour to them
        mask, class_id = extractMask(bboxes, i, masks, roi)

        colour_array = createRandomColourArray(num_detected_objects)
        colour = colour_array[int(class_id)]
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(colour[0]), int(colour[1]), int(colour[2])))

    cv2.imwrite("coloured_blob.png", empty_img)
    cv2.imwrite("object_detection.png", img)

    cv2.imshow("Coloured Blob image", empty_img)
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)