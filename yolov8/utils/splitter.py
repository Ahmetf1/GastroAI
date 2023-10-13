import cv2
import os
import json

full_dataset_dir = '/Users/afa/Desktop/Projects/gastroai/datasets/segmented-images'
train_size = 0.8
validation_size = 0.1
test_size = 0.1
image_height = 512
image_width = 512
class_mapping = {'polyp': 0}

def copy_files(files, input_dir, output_dir):
    for f in files:
        os.system('cp ' + input_dir + '/' + f + ' ' + output_dir + '/' +f)

def resize_files(files, read_dir, write_dir):
    for f in files:
        img = cv2.imread(read_dir + '/' + f)
        img = cv2.resize(img, (image_height, image_width))
        cv2.imwrite(write_dir + '/' + f, img)

def convert_to_yolo(input_dir, output_dir, class_mapping, resize=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = json.load(open(input_dir, 'r'))
    yolo_annotations = {}
    for image_id, image_data in data.items():
        height = image_height if resize else image_data['height']
        width = image_width if resize else image_data['width']
        bboxes = image_data['bbox']
        yolo_bboxes = []

        for bbox in bboxes:
            class_id = class_mapping[bbox['label']]
            xmin = int(bbox['xmin'] * width / image_data['width'])
            ymin = int(bbox['ymin'] * height / image_data['height'])
            xmax = int(bbox['xmax'] * width / image_data['width'])
            ymax = int(bbox['ymax'] * height / image_data['height'])

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_bboxes.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")


        with open(os.path.join(output_dir, f"{image_id}.txt"), 'w') as file:
                file.write("\n".join(yolo_bboxes))


if __name__ == '__main__':
    convert_to_yolo(full_dataset_dir + '/bounding-boxes.json', full_dataset_dir + '/annotations', class_mapping)
    
    os.system('mkdir ' + full_dataset_dir + '/train')
    os.system('mkdir ' + full_dataset_dir + '/test')
    os.system('mkdir ' + full_dataset_dir + '/validation')

    files = os.listdir(full_dataset_dir + '/images')

    train_files = files[:int(len(files) * train_size)]
    validation_files = files[int(len(files) * train_size):int(len(files) * (train_size + validation_size))]
    test_files = files[int(len(files) * (train_size + validation_size)):]

    # resize_files(train_files, full_dataset_dir + '/images', full_dataset_dir + '/train')
    # resize_files(validation_files, full_dataset_dir + '/images', full_dataset_dir + '/validation')
    # resize_files(test_files, full_dataset_dir + '/images', full_dataset_dir + '/test')

    train_labels = list(map(lambda x: x.split('.')[0] + '.txt', train_files))
    validation_labels = list(map(lambda x: x.split('.')[0] + '.txt', validation_files))
    test_labels = list(map(lambda x: x.split('.')[0] + '.txt', test_files))

    copy_files(train_labels, full_dataset_dir + '/annotations', full_dataset_dir + '/train')
    copy_files(validation_labels, full_dataset_dir + '/annotations', full_dataset_dir + '/validation')
    copy_files(test_labels, full_dataset_dir + '/annotations', full_dataset_dir + '/test')


