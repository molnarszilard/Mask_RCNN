"""
Mask R-CNN
Train on the toy Vine dataset and implement color splash effect.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 vine.py train --dataset=/path/to/vine/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 vine.py train --dataset=/path/to/vine/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 vine.py train --dataset=/path/to/vine/dataset --weights=imagenet

    # Apply color splash to an image
    python3 vine.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 vine.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.transform import rescale, resize, downscale_local_mean
import timeit
import cv2
import tensorflow.contrib.slim as slim

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VineConfig(Config):
    """Configuration for training on the vine dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vine"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + vine

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class VineDataset(utils.Dataset):

    def load_vine(self, dataset_dir, subset):
        """Load a subset of the vine dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("vine", 1, "vine")

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir = dataset_dir+"/"

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # Add images
        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']] 

        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     height, width = image.shape[:2]
        #     while len(polygons)<3:
        #         polygons.append(polygons[0])
        #     if len(polygons)>1:
        #         polygons=polygons[:1]
        #     self.add_image(
        #         "vine",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=polygons)
        dlist=os.listdir(dataset_dir)
        dlist.sort()
        for filename in dlist:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image = skimage.io.imread(dataset_dir+filename)
                height, width = image.shape[:2]
                self.add_image(
                    "vine",
                    image_id=filename,  # use file name as a unique image id
                    path=dataset_dir+filename,
                    width=width, height=height,
                    polygons=[])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        folder = "images/"
        if args.cs=="rgb":
            folder = "images"
        elif args.cs=="lab":
            folder = "images_lab"
        elif args.cs=="luv":
            folder = "images_luv"
        elif args.cs=="hls":
            folder = "images_hls"
        elif args.cs=="hsv":
            folder = "images_hsv"
        elif args.cs=="ycrcb":
            folder = "images_ycrcb"
        else:
            print("Unknown color space.")
        image_info = self.image_info[image_id]
        if image_info["source"] != "vine":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        mask = np.zeros([info["height"], info["width"], 1],dtype=np.uint8)
        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1
        # print(mask.shape)
        path = image_info["path"]
        maskgt = cv2.imread(path.replace(folder, 'masks')).astype(np.float32)
        # maskgt = cv2.resize(maskgt,(640,480))
        # maskgt = np.moveaxis(maskgt,-1,0)
        # print(mask.shape)
        maskgt=maskgt[:,:,0]
        mask[maskgt>0]=1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vine":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    path = args.dataset
    if args.cs=="rgb":
        path = path
    elif args.cs=="lab":
        path = path.replace('images', 'images_lab')
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2LAB).astype(np.float32)
    elif args.cs=="luv":
        path = path.replace('images', 'images_luv')
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2LUV).astype(np.float32)
    elif args.cs=="hls":
        path = path.replace('images', 'images_hls')
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2HLS).astype(np.float32)
    elif args.cs=="hsv":
        path = path.replace('images', 'images_hsv')
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV).astype(np.float32)
    elif args.cs=="ycrcb":
        path = path.replace('images', 'images_ycrcb')
        # image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    else:
        print("Unknown color space.")
    dataset_train = VineDataset()
    dataset_train.load_vine(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VineDataset()
    dataset_val.load_vine(args.dataset, "test")
    dataset_val.prepare()
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def darken_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    dark = image/3
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, dark).astype(np.uint8)
    else:
        splash = dark.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = 640
        height = 480
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width, height))

        count = 0
        success = True
        frames = 1
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success and count%frames==0:
                # OpenCV returns images as BGR, convert to RGB
                image = cv2.resize(image, dsize=(640,480), interpolation=cv2.INTER_NEAREST)
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # splash = color_splash(image, r['masks'])
                splash = darken_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
            else:
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def create_mask(model, image_path=None):
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")

    if image_path.endswith('.png') or image_path.endswith('.jpg'):
        image = skimage.io.imread(image_path)
        start = timeit.default_timer()
        r = model.detect([image], verbose=1)[0] #in order to remove the setup-time
        stop = timeit.default_timer()
        setuptime = stop-start
        start = timeit.default_timer()
        r = model.detect([image], verbose=1)[0]
        stop = timeit.default_timer()
        mask = r['masks']
        mask = (np.sum(mask, -1, keepdims=True) >= 1)        
        black = image*0
        white = black+255
        masknorm = np.where(mask, white, black).astype(np.uint8)
        dirname, basename = os.path.split(image_path)
        file_name = args.pred_folder+basename#[:-4]+"_pred_maskrcnn"+'.jpg'
        skimage.io.imsave(file_name, masknorm)
        print('Predicting the image took %f seconds (with setup time: %f)'% (stop-start,setuptime))
    else:
        dlist=os.listdir(image_path)
        dlist.sort()
        time_sum = 0
        wsetuptime = 0
        counter = 0
        for filename in dlist:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                path=image_path+filename
                print("Predicting for:"+filename)
                image = skimage.io.imread(path)
                if counter==0:
                    start = timeit.default_timer()
                    r = model.detect([image], verbose=1)[0] #in order to remove the setup-time
                    stop = timeit.default_timer()
                    setuptime = stop-start
                start = timeit.default_timer()
                r = model.detect([image], verbose=1)[0]
                stop = timeit.default_timer()
                if counter==0:
                    time_sum=stop-start
                    wsetuptime=setuptime
                else:
                    time_sum+=stop-start
                    wsetuptime+=stop-start
                mask = r['masks']
                mask = (np.sum(mask, -1, keepdims=True) >= 1)        
                black = image*0
                white = black+255
                masknorm = np.where(mask, white, black).astype(np.uint8)
                file_name = args.pred_folder+filename#[:-4]+"_pred_maskrcnn"+'.jpg'
                skimage.io.imsave(file_name, masknorm)
                counter+=1
            else:
                continue
        print('Predicting %d images took %f seconds, with the average of %f ( with setup time: %f, average: %f)' % (counter,time_sum,time_sum/counter,wsetuptime,wsetuptime/counter))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vines.')
    parser.add_argument("command",
                        metavar="<command>", default="train",
                        help="'train', 'splash' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="../../dataset/images/",
                        help='Directory of the Vine dataset')
    parser.add_argument('--weights', required=True,
                        default = "coco",
                        metavar="./weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="./logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
    parser.add_argument('--pred_folder', required=False, dest='pred_folder', default='./predicted_images_maskrcnn/', type=str, help='where to save the predicted images.')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VineConfig()
    else:
        class InferenceConfig(VineConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "mask":
        create_mask(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
