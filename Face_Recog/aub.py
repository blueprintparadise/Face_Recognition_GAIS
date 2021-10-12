import matplotlib as mpl
import numpy as np
import seaborn as sns
rcParams = mpl.rcParams
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import albumentations as A
import uuid
import argparse
import dlib
from utils.aux_functions import *

image_directory = r'.\images'

# How Many Augments You Need Per Image
Number_Of_Augments = 4

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
BOX_COLOR = (255, 0, 0)
dir_list = []
dir_list = os.listdir(image_directory)
filtered_files = []
filtered_files_mask = []
image_dir = ''
images = ''


def remove_augmentation():
    for man in dir_list:
        image_dir = image_directory + "\\" + str(man)
        images = os.listdir(image_dir)
        filtered_files = [file for file in images if file.startswith("form_aug")]
        filtered_files_mask = [file for file in images if
                               file.endswith("surgical.jpg") or file.endswith("surgical.png")]
        # print(filtered_files_mask)
        # print(filtered_files)
        for file in filtered_files:
            path_to_file = os.path.join(image_dir, file)
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)
        for filez in filtered_files_mask:
            path_to_file = os.path.join(image_dir, filez)
            if os.path.isfile(path_to_file):
                os.remove(path_to_file)
    return print('Cleared Augmentation')


remove_augmentation()


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def show_image(image, bbox):
    image = visualize_bbox(image.copy(), bbox)
    f = plt.figure(figsize=(18, 12))
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        interpolation='nearest'
    )
    plt.axis('off')
    f.tight_layout()
    plt.show()


def show_augmented(augmentation, image, bbox):
    augmented = augmentation(image=image, bboxes=[bbox], field_id=['1'])
    show_image(augmented['image'], augmented['bboxes'][0])


# [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212].
bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['field_id']
)
# All Possible Augmentations, You can turn them On/Off according to your preference
doc_aug = A.Compose([
    A.Flip(p=0.25),
    A.RandomGamma(gamma_limit=(20, 300), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=35, p=0.9),
    A.RandomRotate90(p=0.25),
    # A.RGBShift(p=0.25),
    # A.GaussNoise(p=0.25),
    # A.RandomToneCurve(p=0.4),
    # A.Solarize(p=0.4, threshold=235),
    # A.CLAHE(p=0.5),
    # A.PiecewiseAffine(p=0.5),
    A.CoarseDropout(p=0.3),
    A.MotionBlur(p=0.5),
    # A.OpticalDistortion(p=0.5),
    A.GridDistortion(p=0.2),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=0.2)

])

dir_list = os.listdir(image_directory)


def masking_images(path):
    # Command-line input setup
    parser = argparse.ArgumentParser(
        description="MaskTheFace - Python code to mask faces dataset"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=path,
        help="Path to either the folder containing images or the image itself",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="surgical",
        choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
        help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Type of the pattern. Available options in masks/textures",
    )

    parser.add_argument(
        "--pattern_weight",
        type=float,
        default=0.5,
        help="Weight of the pattern. Must be between 0 and 1",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="#0473e2",
        help="Hex color value that need to be overlayed to the mask",
    )

    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.5,
        help="Weight of the color intensity. Must be between 0 and 1",
    )

    parser.add_argument(
        "--code",
        type=str,
        # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
        default="",
        help="Generate specific formats",
    )

    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
    )
    parser.add_argument(
        "--write_original_image",
        dest="write_original_image",
        action="store_true",
        help="If true, original image is also stored in the masked folder",
    )
    parser.set_defaults(feature=False)

    args = parser.parse_args()
    args.write_path = args.path + "masked"

    # Set up dlib face detector and predictor
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()

    args.predictor = dlib.shape_predictor(path_to_dlib_model)

    # Extract data from code
    mask_code = "".join(args.code.split()).split(",")
    args.code_count = np.zeros(len(mask_code))
    args.mask_dict_of_dict = {}

    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        args.mask_dict_of_dict[i] = mask_dict

    # Check if path is file or directory or none
    is_directory, is_file, is_other = check_path(args.path)
    display_MaskTheFace()

    if is_directory:
        path, dirs, files = os.walk(args.path).__next__()
        file_count = len(files)
        dirs_count = len(dirs)
        if len(files) > 0:
            print_orderly("Masking image files", 60)

        # Process files in the directory if any
        for f in tqdm(files):
            image_path = path + "/" + f

            write_path = path
            if not os.path.isdir(write_path):
                os.makedirs(write_path)

            if is_image(image_path):
                # Proceed if file is image
                if args.verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)

                split_path = f.rsplit(".")
                masked_image, mask, mask_binary_array, original_image = mask_image(
                    image_path, args
                )
                for i in range(len(mask)):
                    w_path = (
                            write_path
                            + "/"
                            + split_path[0]
                            + "_"
                            + mask[i]
                            + "."
                            + split_path[1]
                    )
                    img = masked_image[i]
                    cv2.imwrite(w_path, img)

        print_orderly("Masking image directories", 60)

        # Process directories withing the path provided
        for d in tqdm(dirs):
            dir_path = args.path + "/" + d
            dir_write_path = args.write_path + "/" + d
            if not os.path.isdir(dir_write_path):
                os.makedirs(dir_write_path)
            _, _, files = os.walk(dir_path).__next__()

            # Process each files within subdirectory
            for f in files:
                image_path = dir_path + "/" + f
                if args.verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)
                write_path = dir_write_path
                if is_image(image_path):
                    # Proceed if file is image
                    split_path = f.rsplit(".")
                    masked_image, mask, mask_binary, original_image = mask_image(
                        image_path, args
                    )
                    for i in range(len(mask)):
                        w_path = (
                                write_path
                                + "/"
                                + split_path[0]
                                + "_"
                                + mask[i]
                                + "."
                                + split_path[1]
                        )
                        w_path_original = write_path + "/" + f
                        img = masked_image[i]
                        # Write the masked image
                        cv2.imwrite(w_path, img)
                        if args.write_original_image:
                            # Write the original image
                            cv2.imwrite(w_path_original, original_image)

                if args.verbose:
                    print(args.code_count)

    # Process if the path was a file
    elif is_file:
        print("Masking image file")
        image_path = args.path
        write_path = args.path.rsplit(".")[0]
        if is_image(image_path):
            # Proceed if file is image
            # masked_images, mask, mask_binary_array, original_image
            masked_image, mask, mask_binary_array, original_image = mask_image(
                image_path, args
            )
            for i in range(len(mask)):
                w_path = write_path + "_" + mask[i] + "." + args.path.rsplit(".")[1]
                img = masked_image[i]
                cv2.imwrite(w_path, img)
    else:
        print("Path is neither a valid file or a valid directory")
    print("Processing Done")


def remove_multimask():
    for man in dir_list:
        image_dir = image_directory + "\\" + str(man)
        images = os.listdir(image_dir)
        # filtered_files = [file for file in images if file.startswith("surgical_surgical")]
        filtered_files_mask = [file for file in images if
                               file.endswith("surgical_surgical.jpg") or file.endswith("surgical_surgical.png")]
        # print(filtered_files_mask)
        for file in filtered_files_mask:
            path_to_file = os.path.join(image_dir, file)
            os.remove(path_to_file)


def process(dir_list):
    for man in (dir_list):
        image_dir = image_directory + "\\" + str(man)
        images = os.listdir(image_dir)

        for i, image_name in enumerate(images):
            img_dir = image_directory + "\\" + str(man) + "/" + image_name
            form = cv2.imread(img_dir)
            print(i, img_dir)
            # STUDENT_ID_BBOX = [500, 600, 400, 400]
            # show_image(form, bbox=STUDENT_ID_BBOX)
            # show_augmented(doc_aug, form, STUDENT_ID_BBOX)
            DATASET_PATH = 'data/augmented'
            IMAGES_PATH = f'{image_dir}'

            os.makedirs(DATASET_PATH, exist_ok=True)
            os.makedirs(IMAGES_PATH, exist_ok=True)

            rows = []
            for i in tqdm(range(Number_Of_Augments)):
                try:
                    augmented = doc_aug(image=form, field_id=['1'])
                    masking_images(IMAGES_PATH)
                except:
                    print("\nSkipped :")
                # print(form)
                file_name = ('form_aug_{}.jpg'.format(uuid.uuid1()))
                # print(image_name+" Done")

                cv2.imwrite(f'{IMAGES_PATH}/{file_name}', augmented['image'])
                remove_multimask()
                # print(filtered_files_mask)



process(dir_list)

