import numpy as np
import os
import cv2
import sys
import tensorflow as tf
import time
import math
import pandas as pd
import random
from scipy import fftpack, signal
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.insert(0, '/home/wayneedham/dependencies/tensorflow/models/research')
# sys.path.insert(0, '/Volumes/Work/MyProject/Django/Chong/models/research')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 100, 'Display logs per step.')

all_columns_below = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'bs1', 'bs2', 'bs3', 'bs4', 'bs5', 'bs6', 'bbs1', 'bbs2',
                     'bbs3', 'bbs4', 'bbs5', 'bbs6']
all_columns_upper = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'us1', 'us2', 'us3', 'us4', 'us5', 'us6', 'uss1', 'uss2',
                     'uss3', 'uss4', 'uss5', 'uss6']


def initialize_detector():
    print('Detect Path %s' % os.getcwd())
    # Name of the directory containing the object detection module we're using
    MODEL_PATH_BELOW = 'model/below-teeth'
    MODEL_PATH_SIDE = 'model/side-teeth'
    MODEL_PATH_UPPER = 'model/upper-teeth'
    MODEL_PATH_BOX_BELOW = 'model/box/below'
    MODEL_PATH_BOX_UPPER = 'model/box/upper'
    MODEL_PATH_BOX_SIDE = 'model/box/side'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    # PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_CKPT_BELOW = os.path.join(CWD_PATH, MODEL_PATH_BELOW, 'frozen_inference_graph.pb')
    PATH_TO_CKPT_SIDE = os.path.join(CWD_PATH, MODEL_PATH_SIDE, 'frozen_inference_graph.pb')
    PATH_TO_CKPT_UPPER = os.path.join(CWD_PATH, MODEL_PATH_UPPER, 'frozen_inference_graph.pb')
    PATH_TO_CKPT_BOX_BELOW = os.path.join(CWD_PATH, MODEL_PATH_BOX_BELOW, 'below-bb-detection.pb')
    PATH_TO_CKPT_BOX_UPPER = os.path.join(CWD_PATH, MODEL_PATH_BOX_UPPER, 'upper-bb-detection.pb')
    PATH_TO_CKPT_BOX_SIDE = os.path.join(CWD_PATH, MODEL_PATH_BOX_SIDE, 'side-bb-detection.pb')

    # Path to label map file
    PATH_TO_LABELS_BELOW = os.path.join(CWD_PATH, MODEL_PATH_BELOW, 'horse-detection.pbtxt')
    PATH_TO_LABELS_SIDE = os.path.join(CWD_PATH, MODEL_PATH_SIDE, 'side-detection.pbtxt')
    PATH_TO_LABELS_UPPER = os.path.join(CWD_PATH, MODEL_PATH_UPPER, 'upper-detection.pbtxt')
    PATH_TO_LABELS_BOX_BELOW = os.path.join(CWD_PATH, MODEL_PATH_BOX_BELOW, 'below-bb-detection.pbtxt')
    PATH_TO_LABELS_BOX_UPPER = os.path.join(CWD_PATH, MODEL_PATH_BOX_UPPER, 'upper-bb-detection.pbtxt')
    PATH_TO_LABELS_BOX_SIDE = os.path.join(CWD_PATH, MODEL_PATH_BOX_SIDE, 'side-bb-detection.pbtxt')

    # Path to image

    # Number of classes the object detector can identify
    NUM_CLASSES_BELOW = 18
    NUM_CLASSES_SIDE = 12
    NUM_CLASSES_UPPER = 18
    NUM_CLASSES_BOX_BELOW = 1
    NUM_CLASSES_BOX_UPPER = 1
    NUM_CLASSES_BOX_SIDE = 1

    label_map_below = label_map_util.load_labelmap(PATH_TO_LABELS_BELOW)
    label_map_side = label_map_util.load_labelmap(PATH_TO_LABELS_SIDE)
    label_map_upper = label_map_util.load_labelmap(PATH_TO_LABELS_UPPER)
    label_map_box_below = label_map_util.load_labelmap(PATH_TO_LABELS_BOX_BELOW)
    label_map_box_upper = label_map_util.load_labelmap(PATH_TO_LABELS_BOX_UPPER)
    label_map_box_side = label_map_util.load_labelmap(PATH_TO_LABELS_BOX_SIDE)

    categories_below = label_map_util.convert_label_map_to_categories(label_map_below,
                                                                      max_num_classes=NUM_CLASSES_BELOW,
                                                                      use_display_name=True)
    categories_side = label_map_util.convert_label_map_to_categories(label_map_side, max_num_classes=NUM_CLASSES_SIDE,
                                                                     use_display_name=True)
    categories_upper = label_map_util.convert_label_map_to_categories(label_map_upper,
                                                                      max_num_classes=NUM_CLASSES_UPPER,
                                                                      use_display_name=True)
    categories_box_below = label_map_util.convert_label_map_to_categories(label_map_box_below,
                                                                          max_num_classes=NUM_CLASSES_BOX_BELOW,
                                                                          use_display_name=True)
    categories_box_upper = label_map_util.convert_label_map_to_categories(label_map_box_upper,
                                                                          max_num_classes=NUM_CLASSES_BOX_UPPER,
                                                                          use_display_name=True)
    categories_box_side = label_map_util.convert_label_map_to_categories(label_map_box_side,
                                                                         max_num_classes=NUM_CLASSES_BOX_SIDE,
                                                                         use_display_name=True)

    global category_index_below
    category_index_below = label_map_util.create_category_index(categories_below)
    global detection_graph_below
    detection_graph_below = tf.Graph()
    with detection_graph_below.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_BELOW, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    global category_index_side
    category_index_side = label_map_util.create_category_index(categories_side)
    global detection_graph_side
    detection_graph_side = tf.Graph()
    with detection_graph_side.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_SIDE, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    global category_index_upper
    category_index_upper = label_map_util.create_category_index(categories_upper)
    global detection_graph_upper
    detection_graph_upper = tf.Graph()
    with detection_graph_upper.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_UPPER, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    global category_index_box_below
    category_index_box_below = label_map_util.create_category_index(categories_box_below)
    global detection_graph_box_below
    detection_graph_box_below = tf.Graph()
    with detection_graph_box_below.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_BOX_BELOW, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    global category_index_box_upper
    category_index_box_upper = label_map_util.create_category_index(categories_box_upper)
    global detection_graph_box_upper
    detection_graph_box_upper = tf.Graph()
    with detection_graph_box_upper.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_BOX_UPPER, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    global category_index_box_side
    category_index_box_side = label_map_util.create_category_index(categories_box_side)
    global detection_graph_box_side
    detection_graph_box_side = tf.Graph()
    with detection_graph_box_side.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_BOX_SIDE, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    data = np.array(image.getdata())

    return data[:, :3].reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def feature_normalize(train_X):
    global mean, std
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    return (train_X - mean) / std


def run_training(train_X, train_Y, age_data):
    m = len(train_X)
    n = len(train_X[0])

    X = tf.placeholder(tf.float32, [m, n])
    Y = tf.placeholder(tf.float32, [m, 1])

    # weights
    W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

    # linear model
    activation = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_sum(tf.square(activation - Y)) / (2 * m)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            sess.run(optimizer, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})

            if step % FLAGS.display_step == 0:
                print("Step:", "%04d" % (step + 1), "Cost=",
                      "{:.2f}".format(sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})),
                      "W=",
                      sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})
        print("Training Cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        count = len(age_data)
        x_data = np.array(age_data, dtype=np.float32).reshape(1, count)
        predict_X = (x_data - mean) / std
        predict_y = tf.add(tf.matmul(predict_X, W), b)
        age = sess.run(predict_y)

        return float(age)


def ageDetect(model_path, age_data):
    count = len(age_data)
    tf.reset_default_graph()
    # weights
    W = tf.get_variable("weight", [count, 1], dtype=np.float32)
    b = tf.get_variable("bias", [1], dtype=np.float32)

    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, model_path)

        # graph = tf.get_default_graph()
        #
        #
        # W = tf.get_variable("weight", [18, 1],
        #                     initializer=tf.random_normal_initializer())
        # b = tf.get_variable("bias", [1],
        #                     initializer=tf.constant_initializer(0.0))

        x_data = np.array(age_data, dtype=np.float32).reshape(1, count)
        predict_X = (x_data - mean) / std
        predict_y = tf.add(tf.matmul(predict_X, W), b)
        age = session.run(predict_y)

        # age = sess.run(cost, feed_dict={X: x_data})

        print("Detected Age: %f" % age)

        return float(age)


def filter_duplicated_labels(labels, scores):
    # labels = ['u1','b1','u2','u1','u1','b1','u1','u2','b1','b2','b4','b3']
    # scores = [0.9,  0.5, 0.8, 0.4, 0.7, 1.0, 0.3, 0.5, 0.8, 0.7, 0.95,0.92]
    index = 0
    duplicated_indexes = []
    while index < len(labels):
        label = labels[index]

        if index in duplicated_indexes:
            index = index + 1
            continue

        max_score = scores[index]
        max_index = index

        for i in range(index + 1, len(labels)):
            if i in duplicated_indexes:
                continue

            sec_label = labels[i]
            sec_score = scores[i]

            if label == sec_label:
                if sec_score > max_score:
                    duplicated_indexes.append(max_index)
                    max_score = sec_score
                    max_index = i
                else:
                    duplicated_indexes.append(i)

        index = index + 1

    reordered = sorted(duplicated_indexes, key=int, reverse=True)

    return reordered


def get_recogized_columns(arr1, arr2):
    result = []
    for column in arr2:
        if column in arr1:
            result.append(column)
    return result


def get_unrecogized_columns(arr1, arr2):
    result = []
    for column in arr2:
        if column not in arr1:
            result.append(column)
    return result


def contain_array(arr1, arr2):
    for column in arr2:
        if column not in arr1:
            return False
    return True


def detect_age_below(imgPath):
    image = Image.open(imgPath)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_below)
    # Visualization of the results of a detection.
    label_dict = []
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] >= 0.9:
            item_dict = {'box': output_dict['detection_boxes'][i], 'score': output_dict['detection_scores'][i],
                         'class': category_index_below[output_dict['detection_classes'][i]]}
            label_dict.append(item_dict)

    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    model_columns = []
    age_data = []
    scores = []
    boxes = []
    classes = []
    for item in label_dict:
        box = item['box']
        boxes.append(box)
        item_class = item['class']
        classes.append(item_class['id'])
        score = item['score']
        scores.append(score)
        id = item_class['name']
        model_columns.append(id)
        ymin, xmin, ymax, xmax = box
        item_ratio = ((float(ymax) - float(ymin)) * float(img_height)) / (
                float(img_width) * (float(xmax) - float(xmin)))
        age_data.append(item_ratio)

    reordered_indexes = filter_duplicated_labels(model_columns, scores)
    for i in reordered_indexes:
        del model_columns[i]
        del age_data[i]
        del scores[i]
        del boxes[i]
        del classes[i]

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(boxes),
        np.array(classes),
        np.array(scores),
        category_index_below,
        min_score_thresh=.9,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        return_label=label_dict,
        draw=True)

    train_data = pd.read_csv('model/below-teeth/horse_below.csv')
    x_data = []
    y_data = train_data['age'].tolist()

    for column in model_columns:
        data = train_data[column].tolist()
        x_data.append(data)

    train_Y = np.array(y_data).astype('float32')
    train_X = np.array(x_data).astype('float32')
    train_X = np.transpose(train_X)
    train_Y = train_Y.reshape(len(train_Y), 1)
    train_X = feature_normalize(train_X)

    model_path = 'model/below-teeth/age-model'
    age = 0
    with tf.Graph().as_default():
        age = run_training(train_X, train_Y, age_data)
    print('----below-----', age)
    # age = ageDetect(age_model_path, age_data)

    ts = time.time()
    im_index = int(ts * 1000)
    out_name = 'output-' + str(im_index) + '.jpg'
    out_path = 'media/' + out_name
    # out_path = os.path.join(OUTPUT_DIR, out_name)

    dst = cv2.resize(image_np, (int(img_width), int(img_height)))

    img = Image.fromarray(dst, 'RGB')
    img.save(imgPath)
    # img.save(out_path)
    bs_columns = ['bs1', 'bs2', 'bs3', 'bs4', 'bs5', 'bs6']
    recognized_bs_columns = get_recogized_columns(model_columns, bs_columns)
    if 'bs3' not in recognized_bs_columns and 'bs4' not in recognized_bs_columns and len(recognized_bs_columns) > 2:
        # age = 15
        age = random.randint(14, 16)
    if 'bs2' not in recognized_bs_columns and 'bs3' not in recognized_bs_columns and 'bs4' not in recognized_bs_columns\
            and 'bs5' not in recognized_bs_columns:
        # age = 16
        age = random.randint(15, 17)
    if len(recognized_bs_columns) == 2 or len(recognized_bs_columns) == 1:
        age = 16
    if len(recognized_bs_columns) == 0 or age > 17:
        age = 17
    return age


def detect_age_upper(imgPath):
    image = Image.open(imgPath)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_upper)
    # Visualization of the results of a detection.
    label_dict = []
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] >= 0.9:
            item_dict = {'box': output_dict['detection_boxes'][i], 'score': output_dict['detection_scores'][i],
                         'class': category_index_upper[output_dict['detection_classes'][i]]}
            label_dict.append(item_dict)

    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    model_columns = []
    age_data = []
    scores = []
    boxes = []
    classes = []
    for item in label_dict:
        box = item['box']
        boxes.append(box)
        item_class = item['class']
        classes.append(item_class['id'])
        score = item['score']
        scores.append(score)
        id = item_class['name']
        model_columns.append(id)
        ymin, xmin, ymax, xmax = box
        item_ratio = ((float(ymax) - float(ymin)) * float(img_height)) / (
                float(img_width) * (float(xmax) - float(xmin)))
        age_data.append(item_ratio)

    reordered_indexes = filter_duplicated_labels(model_columns, scores)
    for i in reordered_indexes:
        del model_columns[i]
        del age_data[i]
        del scores[i]
        del boxes[i]
        del classes[i]

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(boxes),
        np.array(classes),
        np.array(scores),
        category_index_upper,
        min_score_thresh=.9,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        return_label=label_dict,
        draw=True)

    train_data = pd.read_csv('model/upper-teeth/horse_upper.csv')
    x_data = []
    y_data = train_data['age'].tolist()

    for column in model_columns:
        data = train_data[column].tolist()
        x_data.append(data)

    train_Y = np.array(y_data).astype('float32')
    train_X = np.array(x_data).astype('float32')
    train_X = np.transpose(train_X)
    train_Y = train_Y.reshape(len(train_Y), 1)
    train_X = feature_normalize(train_X)

    model_path = 'model/below-teeth/age-model'
    age = 0
    with tf.Graph().as_default():
        age = run_training(train_X, train_Y, age_data)
    print('---upper---', age)
    # age = ageDetect(age_model_path, age_data)

    ts = time.time()
    im_index = int(ts * 1000)
    out_name = 'output-' + str(im_index) + '.jpg'
    out_path = 'media/' + out_name
    # out_path = os.path.join(OUTPUT_DIR, out_name)

    dst = cv2.resize(image_np, (int(img_width), int(img_height)))

    img = Image.fromarray(dst, 'RGB')
    img.save(imgPath)
    # img.save(out_path)
    bs_columns = ['us1', 'us2', 'us3', 'us4', 'us5', 'us6']
    recognized_bs_columns = get_recogized_columns(model_columns, bs_columns)
    is_pass = True
    if 'us2' not in recognized_bs_columns and 'us3' not in recognized_bs_columns and 'us4' not in recognized_bs_columns\
            and 'us5' not in recognized_bs_columns:
        is_pass = False
        # age = 19
        age = random.randint(18, 20)

    if 'us3' not in recognized_bs_columns and 'us4' not in recognized_bs_columns and is_pass:
        # age = 18
        age = random.randint(17, 19)

    if len(recognized_bs_columns) == 0 or age > 20:
        age = 20
    return age


def detect_age_side(imgPath):
    image = Image.open(imgPath)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_side)
    # Visualization of the results of a detection.
    label_dict = []
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_side,
        min_score_thresh=.9,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        return_label=label_dict,
        draw=True)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    model_columns = []
    scores = []
    age_data = []
    model_type = 0

    for item in label_dict:
        box = item['box']
        item_class = item['class']
        id = item_class['name']
        score = item['score']
        scores.append(score)
        model_columns.append(id)
        ymin, xmin, ymax, xmax = box
        item_ratio = ((float(ymax) - float(ymin)) * float(img_height)) / (
                float(img_width) * (float(xmax) - float(xmin)))
        age_data.append(item_ratio)

    left_labels = ['u1', 'b1', 'u2', 'b2', 'ug1', 'ug2']
    right_labels = ['u5', 'b5', 'u6', 'b6', 'ug5', 'ug6']
    if label_dict[0]['class']['name'] in left_labels:
        model_type = 0
    else:
        model_type = 1

    error = False
    if len(model_columns) == 0:
        error = True
    if model_type == 0:
        bad_indexes = []
        for i in range(0, len(model_columns)):
            if model_columns[i] in right_labels:
                bad_indexes.append(i)
        reordered_bad_indexes = sorted(bad_indexes, key=int, reverse=True)
        for i in reordered_bad_indexes:
            del model_columns[i]
            del age_data[i]
            del scores[i]
    else:
        bad_indexes = []
        for i in range(0, len(model_columns)):
            if model_columns[i] in left_labels:
                bad_indexes.append(i)
        reordered_bad_indexes = sorted(bad_indexes, key=int, reverse=True)
        for i in reordered_bad_indexes:
            del model_columns[i]
            del age_data[i]
            del scores[i]

    if error == True:
        return None

    reordered_indexes = filter_duplicated_labels(model_columns, scores)
    for i in reordered_indexes:
        del model_columns[i]
        del age_data[i]
        del scores[i]

    train_path = ''
    if model_type == 0:
        train_path = 'model/side-teeth/horse_side_l.csv'
    else:
        train_path = 'model/side-teeth/horse_side_r.csv'

    train_data = pd.read_csv(train_path)
    x_data = []
    y_data = train_data['age'].tolist()

    for column in model_columns:
        data = train_data[column].tolist()
        x_data.append(data)

    train_Y = np.array(y_data).astype('float32')
    train_X = np.array(x_data).astype('float32')
    train_X = np.transpose(train_X)
    train_Y = train_Y.reshape(len(train_Y), 1)
    train_X = feature_normalize(train_X)

    model_path = 'side-teeth/age-model'
    age = 0
    with tf.Graph().as_default():
        age = run_training(train_X, train_Y, age_data)
    print(age)
    # age = ageDetect(age_model_path, age_data)

    ts = time.time()
    im_index = int(ts * 1000)
    out_name = 'output-' + str(im_index) + '.jpg'
    out_path = 'media/' + out_name
    # out_path = os.path.join(OUTPUT_DIR, out_name)

    dst = cv2.resize(image_np, (int(img_width), int(img_height)))

    img = Image.fromarray(dst, 'RGB')
    img.save(imgPath)

    return age


def detect_box_below(imgpath):
    root_dir = 'media/detect_pics'
    suffix = os.path.splitext(imgpath)[1]
    filename = os.path.basename(imgpath).split(suffix)[0]
    image = Image.open(imgpath)

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_box_below)
    # Visualization of the results of a detection.
    label_dict = []
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_box_below,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=1,
        return_label=label_dict,
        draw=False)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    item = label_dict[0]
    ymin, xmin, ymax, xmax = item['box']

    ymin = ymin * img_height
    xmin = xmin * img_width
    xmax = xmax * img_width
    ymax = ymax * img_height

    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    name = 'img_' + str(int(time.time() * 1000)) + suffix
    out_path = os.path.join(root_dir, name)
    # img = Image.fromarray(cropped_image, 'RGB')
    cropped_image.save(out_path)

    return out_path


def detect_box_upper(imgpath):
    root_dir = 'media/detect_pics'
    suffix = os.path.splitext(imgpath)[1]
    filename = os.path.basename(imgpath).split(suffix)[0]
    image = Image.open(imgpath)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_box_upper)
    # Visualization of the results of a detection.
    label_dict = []
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_box_upper,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=1,
        return_label=label_dict,
        draw=False)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    item = label_dict[0]
    ymin, xmin, ymax, xmax = item['box']

    ymin = ymin * img_height
    xmin = xmin * img_width
    xmax = xmax * img_width
    ymax = ymax * img_height

    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    name = 'img_' + str(int(time.time() * 1000)) + suffix
    out_path = os.path.join(root_dir, name)
    # img = Image.fromarray(cropped_image, 'RGB')
    cropped_image.save(out_path)

    return out_path


def detect_box_side(imgpath):
    root_dir = 'media/detect_pics'
    suffix = os.path.splitext(imgpath)[1]
    image = Image.open(imgpath)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph_box_side)
    # Visualization of the results of a detection.
    label_dict = []
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_box_side,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=1,
        return_label=label_dict,
        draw=False)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    if len(label_dict) == 0:
        return None
    img_width = image.size[0]
    img_height = image.size[1]

    item = label_dict[0]
    ymin, xmin, ymax, xmax = item['box']

    ymin = ymin * img_height
    xmin = xmin * img_width
    xmax = xmax * img_width
    ymax = ymax * img_height

    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    name = 'img_' + str(int(time.time() * 1000)) + suffix
    out_path = os.path.join(root_dir, name)
    # img = Image.fromarray(cropped_image, 'RGB')
    cropped_image.save(out_path)

    return out_path

# Test
# initialize_detector()

# filter_duplicated_labels([],[])
