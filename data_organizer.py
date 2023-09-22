from pathlib import Path
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
from sklearn.utils import shuffle
import albumentations as A
from PIL import Image
import numpy as np


def add_augmentations(Path_input_train_only_images:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\train"),Path_input_train_only_labels:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\train"),path_output_images:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\train_with_augmentations"),path_output_labels:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\train_with_augmentations")):
    list_of_all_images = list(Path_input_train_only_images.glob('*.png'))
    list_of_all_images = sorted(list_of_all_images, key=lambda x: x.name.split('.')[0])
    list_of_all_text_files = sorted(list(Path_input_train_only_labels.glob('*.txt')), key=lambda x: x.name.split('.')[0])
    zipped_image_and_txt = list(zip(list_of_all_images,list_of_all_text_files))
    transform_RGBShift = A.Compose([A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, always_apply=False, p=1)])
    transform_CLAHE = A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1)])
    transform_RandomBrightness = A.Compose([A.RandomBrightness(limit=0.3, always_apply=False, p=1)])
    transform_RandomGamma = A.Compose([A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=1)])
    # transform_Blur = A.Compose([A.Blur(blur_limit=8, always_apply=False, p=1)])
    # transform_MedianBlur = A.Compose([A.MedianBlur(blur_limit=7, always_apply=False, p=1)])
    transforms_GaussianBlur  = A.Compose([A.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=1)])
    transform_JpegCompression = A.Compose([A.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=1)])
    transform_Sharpen = A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1)])
    # transform_RandomShadow = A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1)])
    transform_ISOnoise = A.Compose([A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1)])

    list_all_transforms = [transform_RGBShift,transform_CLAHE,transform_RandomBrightness,transform_RandomGamma,transforms_GaussianBlur,transform_JpegCompression,transform_Sharpen,transform_ISOnoise]
    
    for image_and_text in tqdm(zipped_image_and_txt, total=len(zipped_image_and_txt)):
        image = cv2.imread(image_and_text[0].as_posix())
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        list_of_transformed_images_including_original_image = []
        for transform in list_all_transforms:
            transformed = transform(image=image)
            transformed_image = transformed['image']
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            list_of_transformed_images_including_original_image.append(transformed_image)

        #saving original image
        original_image_name = Path(path_output_images.as_posix() + '/' + image_and_text[0].name)
        original_image_txt = Path(path_output_labels.as_posix() + '/' + image_and_text[1].name)
        cv2.imwrite(original_image_name.as_posix(),original_image)
        shutil.copyfile(image_and_text[1], original_image_txt)

        #saving augmentations
        for index,image in enumerate(list_of_transformed_images_including_original_image):
            image_name = Path(path_output_images.as_posix() + '/' + image_and_text[0].name.split('.')[0]+f'_AUGMENTATION_{index+1}.png')
            txt_file_name = Path(path_output_labels.as_posix() + '/' + image_and_text[1].name.split('.')[0]+f'_AUGMENTATION_{index+1}.txt')
            cv2.imwrite(image_name.as_posix(),image)
            shutil.copyfile(image_and_text[1], txt_file_name)
            # cv2.imshow('test', image)
            # cv2.waitKey(1)



def train_test_split_mine(input_path:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\one group data\images"),
                     train_path:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\train"),
                     val_path:Path=Path(r"C:\Coding projects\yolo\data\ready_to_go_data\train_test_split\val")):
    
    list_of_images_paths_sorted = sorted(list(input_path.rglob('*.png'))+list(input_path.glob('*.jpg')))
    list_of_text_files_annotations = sorted(list(input_path.parent.rglob('*.txt')))
    zipped_list= list(zip(list_of_images_paths_sorted,list_of_text_files_annotations))
    train, val = train_test_split(zipped_list, random_state=42, test_size=0.1)
    for item in tqdm(train, total=len(train)):
        shutil.copyfile(item[0].as_posix(), Path(train_path.as_posix() + '/' + item[0].name).as_posix())
        shutil.copyfile(item[1].as_posix(), Path(train_path.as_posix() + '/' + item[1].name).as_posix())
    for item in tqdm(val, total=len(val)):
        shutil.copyfile(item[0].as_posix(), Path(val_path.as_posix() + '/' + item[0].name).as_posix())
        shutil.copyfile(item[1].as_posix(), Path(val_path.as_posix() + '/' + item[1].name).as_posix())

def extracting_annotations_from_supervisely_for_classifier(ann_path:Path)->dict:
    a= 2

def get_intersection():
    scoreboards = Path(r"/media/access/New Volume/scoreboard_data_25_07_2022/Scoreboards (Batch-2)")
    words_and_letter_path1 = Path(r"/media/access/New Volume/scoreboard_data_25_07_2022/Words & Letters 1 (Batch-2)")
    words_and_letter_path2 = Path(r"/media/access/New Volume/scoreboard_data_25_07_2022/Words & Letters 2 (Batch-2)")
    all_images_from_scoreboards = []
    for img_dir in [Path(it.as_posix() + '/img') for it in list(scoreboards.iterdir())[:-1]]:
        all_images_from_scoreboards += [it.name for it in list(img_dir.glob('*.png'))]
    all_images_from_words_and_letters = []
    for img_dir in [Path(it.as_posix() + '/img') for it in list(words_and_letter_path1.iterdir())[:-1]] + [Path(it.as_posix() + '/img') for it in list(words_and_letter_path2.iterdir())[:-1]] :
        all_images_from_words_and_letters += [it.name.replace('.jpg', '') for it in list(img_dir.glob('*.png'))]
    intersection_of_images = list(set(all_images_from_words_and_letters) & set(all_images_from_scoreboards))
    len_of_intersection = len(intersection_of_images)
    return intersection_of_images

def Data_validator(output_resolution,data_path:Path, sample_size):
    # ziped_list_of_images_and_files = list(zip(sorted(list(data_path.glob('*.png'))+list(data_path.glob('*.jpg'))), sorted(list(data_path.glob('*.txt')))))
    res = []
    list_of_txt_files_paths_that_contains_annotations = sorted([it for it in list(data_path.rglob('*.txt'))])
    list_of_images_paths_that_contains_scoreboards = sorted([it for it in list(data_path.rglob('*.png'))+list(data_path.rglob('*.jpg'))])

    shuffled_list = shuffle(list(zip(list_of_txt_files_paths_that_contains_annotations,list_of_images_paths_that_contains_scoreboards)))[:sample_size]
    list_of_txt_files_to_run_on = [it[0] for it in shuffled_list]
    list_of_images_to_run_on = [it[1] for it in shuffled_list]
    
    for annotation_txt_file_path in tqdm(list_of_txt_files_to_run_on,total=len(list_of_txt_files_to_run_on)):
        annotation_file_name = annotation_txt_file_path.name.split('.')[0]
        for image_path in list_of_images_paths_that_contains_scoreboards:
            if annotation_file_name == image_path.name.split('.')[0]:
                try:
                    with open(annotation_txt_file_path.as_posix(), 'r') as file:
                        annotation = eval('['+file.readline().replace(' ', ',')+']')
                        res.append((image_path,annotation))
                        file.close()
                except:
                    print(f"failed with: {annotation_file_name}")
    for item in res:
        img_pil = Image.open(item[0].as_posix())
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, output_resolution)
        center_x = item[1][1]
        center_y = item[1][2]
        width = item[1][3]
        height = item[1][4]
        tl_x = center_x - 0.5*width
        tl_y = center_y - 0.5*height
        Class = 0
        ###validation
        pt1_x = int((tl_x+width)*img_resized.shape[1])
        pt2_x = int(tl_x*img_resized.shape[1])
        pt1_y = int((tl_y+height)*img_resized.shape[0])
        pt2_y = int(tl_y*img_resized.shape[0])
        cv2.rectangle(img_resized, pt1=(pt1_x,pt1_y), pt2=(pt2_x,pt2_y), color=(0,0,255), thickness=1)
        cv2.circle(img_resized, (int(center_x*img_resized.shape[0]),int(center_y*img_resized.shape[1])), radius=0, color=(0, 255, 0), thickness=5)
        img_resized = cv2.resize(img_resized, (1280,720))
        cv2.imshow('img_resized',img_resized)
        cv2.waitKey(0)


def data_organizer_supervisely_peter_script_output(output_resolution,annotations_dir:Path, images_dir:Path, output_dir_train:Path):
    res = []
    list_of_txt_files_paths_that_contains_annotations = [it for it in list(annotations_dir.rglob('*.txt'))]
    list_of_images_paths_that_contains_scoreboards = [it for it in images_dir.rglob('*.png')]
    for annotation_txt_file_path in tqdm(list_of_txt_files_paths_that_contains_annotations,total=len(list_of_txt_files_paths_that_contains_annotations)):
        annotation_file_name = annotation_txt_file_path.name.split('.')[0]
        for image_path in list_of_images_paths_that_contains_scoreboards:
            if annotation_file_name == image_path.name.split('.')[0]:
                try:
                    with open(annotation_txt_file_path.as_posix(), 'r') as file:
                        annotation = eval('['+file.readline().replace(' ', ',')+']')
                        res.append((image_path,annotation))
                        file.close()
                except:
                    print(f"failed with: {annotation_file_name}")
    
    #### saving new resized images
    #### saving two files in each iteration:
    # <name>.jpg - resized image with resolution of 416x416
    # <name>.txt - txt file that contains the the class (int) center of the annotation (two floats) rectangle and the width and height (two floats) of it in this format: <object-class> <x_center> <y_center> <width> <height>
    for item in [it for it in res if it[1]!=[]]:
        img = cv2.imread(item[0].as_posix())
        img_resized_org = cv2.resize(img, output_resolution)
        img_resized = img_resized_org.copy()
        center_x = item[1][1]
        center_y = item[1][2]
        width = item[1][3]
        height = item[1][4]
        left_increase_param = 0.5*width
        right_increase_param = 0.5*width
        up_increase_param = 0.5*height
        down_increase_param = 0.5*height
        Class = 0
        ##validation
        # while int((center_x-left_increase_param)*img_resized.shape[1]) < 0:
        #     left_increase_param = 0.95*left_increase_param
        # while int((center_x-left_increase_param)*img_resized.shape[1]) > 416:
        #     left_increase_param = 1.05*left_increase_param
        
        # while int((center_x+right_increase_param)*img_resized.shape[1]) < 0:
        #     right_increase_param = 0.95*right_increase_param
        # while int((center_x+right_increase_param)*img_resized.shape[1]) > 416:
        #     right_increase_param = 1.05*right_increase_param

        # while int((center_y+up_increase_param)*img_resized.shape[1]) < 0:
        #     up_increase_param = 1.05*up_increase_param
        # while int((center_y+up_increase_param)*img_resized.shape[1]) > 416:
        #     up_increase_param = 0.95*up_increase_param

        # while int((center_y-down_increase_param)*img_resized.shape[1]) < 0:
        #     down_increase_param = 1.05*down_increase_param
        # while int((center_y-down_increase_param)*img_resized.shape[1]) > 416:
        #     down_increase_param = 0.95*down_increase_param

        pt1_x = int((center_x-left_increase_param)*img_resized.shape[1])
        pt2_x = int((center_x+right_increase_param)*img_resized.shape[1])
        pt1_y = int((center_y+up_increase_param)*img_resized.shape[1])
        pt2_y = int((center_y-down_increase_param)*img_resized.shape[1])
            
        cv2.rectangle(img_resized, pt1=(pt1_x,pt1_y), pt2=(pt2_x,pt2_y), color=(0,0,255), thickness=3)
        cv2.circle(img_resized, (int(center_x*img_resized.shape[0]),int(center_y*img_resized.shape[1])), radius=0, color=(0, 255, 0), thickness=5)
        cv2.imshow('img_resized',img_resized)
        cv2.waitKey(1)
        center_x_after_modification = ((center_x-left_increase_param)+(center_x+right_increase_param))/2
        center_y_after_modification = ((center_y+up_increase_param)+(center_y-down_increase_param))/2
        width_after_modification = abs((center_x-left_increase_param)-(center_x+right_increase_param))
        height_after_modification = abs((center_y+up_increase_param)-(center_y-down_increase_param))
        with open(Path(output_dir_train.as_posix() + '/' + item[0].name.split('.')[0] + '.txt').as_posix(), 'w') as f:
            f.write(f'{Class} {center_x_after_modification} {center_y_after_modification} {width_after_modification} {height_after_modification}')
        cv2.imwrite(Path(output_dir_train.as_posix() + '/' + item[0].name.split('.')[0] + '.png').as_posix(),img_resized_org)
    
    


def data_organizer(output_resolution,annotations_dir:Path, images_dir:Path, output_dir_train:Path, output_dir_test:Path):
    res = []
    list_of_txt_files_paths_that_contains_annotations = [it for it in annotations_dir.glob('*.txt')]
    list_of_images_paths_that_contains_scoreboards = [it for it in images_dir.glob('*.png')]
    for annotation_txt_file_path in list_of_txt_files_paths_that_contains_annotations:
        annotation_file_name = annotation_txt_file_path.name.split('.')[0]
        for image_path in list_of_images_paths_that_contains_scoreboards:
            if annotation_file_name == image_path.name.split('.')[0]:
                try:
                    with open(annotation_txt_file_path.as_posix(), 'r') as file:
                        annotation = eval(file.read())
                        res.append((image_path,annotation))
                        file.close()
                except:
                    print(f"failed with: {annotation_file_name}")
    
    #### saving new resized images
    #### saving two files in each iteration:
    # <name>.jpg - resized image with resolution of 416x416
    # <name>.txt - txt file that contains the the class (int) center of the annotation (two floats) rectangle and the width and height (two floats) of it in this format: <object-class> <x_center> <y_center> <width> <height>
    for item in res[15:]:
        img = cv2.imread(item[0].as_posix())
        img_resized_org = cv2.resize(img, output_resolution)
        img_resized = img_resized_org.copy()
        tl_x = item[1][0] / img.shape[1]
        tl_y = item[1][1] / img.shape[0]
        width = item[1][2] / img.shape[1]
        height = item[1][3] / img.shape[0]
        center_x = tl_x + 0.5*width
        center_y = tl_y + 0.5*height
        left_increase_param = 0.5*width
        right_increase_param = 0.5*width
        up_increase_param = 0.5*height
        down_increase_param = 0.5*height
        Class = 0
        ##validation
        # while int((center_x-left_increase_param)*img_resized.shape[1]) < 0:
        #     left_increase_param = 0.95*left_increase_param
        # while int((center_x-left_increase_param)*img_resized.shape[1]) > 416:
        #     left_increase_param = 1.05*left_increase_param
        
        # while int((center_x+right_increase_param)*img_resized.shape[1]) < 0:
        #     right_increase_param = 0.95*right_increase_param
        # while int((center_x+right_increase_param)*img_resized.shape[1]) > 416:
        #     right_increase_param = 1.05*right_increase_param

        # while int((center_y+up_increase_param)*img_resized.shape[1]) < 0:
        #     up_increase_param = 1.05*up_increase_param
        # while int((center_y+up_increase_param)*img_resized.shape[1]) > 416:
        #     up_increase_param = 0.95*up_increase_param

        # while int((center_y-down_increase_param)*img_resized.shape[1]) < 0:
        #     down_increase_param = 1.05*down_increase_param
        # while int((center_y-down_increase_param)*img_resized.shape[1]) > 416:
        #     down_increase_param = 0.95*down_increase_param

        pt1_x = int((center_x-left_increase_param)*img_resized.shape[1])
        pt2_x = int((center_x+right_increase_param)*img_resized.shape[1])
        pt1_y = int((center_y+up_increase_param)*img_resized.shape[1])
        pt2_y = int((center_y-down_increase_param)*img_resized.shape[1])
            
        cv2.rectangle(img_resized, pt1=(pt1_x,pt1_y), pt2=(pt2_x,pt2_y), color=(0,0,255), thickness=3)
        cv2.circle(img_resized, (int(center_x*img_resized.shape[0]),int(center_y*img_resized.shape[1])), radius=0, color=(0, 255, 0), thickness=5)
        cv2.imshow('img_resized',img_resized)
        cv2.waitKey(1)
        center_x_after_modification = ((center_x-left_increase_param)+(center_x+right_increase_param))/2
        center_y_after_modification = ((center_y+up_increase_param)+(center_y-down_increase_param))/2
        width_after_modification = abs((center_x-left_increase_param)-(center_x+right_increase_param))
        height_after_modification = abs((center_y+up_increase_param)-(center_y-down_increase_param))
        with open(Path(output_dir_train.as_posix() + '/' + item[0].name.split('.')[0] + '.txt').as_posix(), 'w') as f:
            f.write(f'{Class} {center_x_after_modification} {center_y_after_modification} {width_after_modification} {height_after_modification}')
        cv2.imwrite(Path(output_dir_train.as_posix() + '/' + item[0].name.split('.')[0] + '.png').as_posix(),img_resized_org)
    
    for item in res[:15]:
        img = cv2.imread(item[0].as_posix())
        img_resized_org = cv2.resize(img, output_resolution)
        img_resized = img_resized_org.copy()
        tl_x = item[1][0] / img.shape[1]
        tl_y = item[1][1] / img.shape[0]
        width = item[1][2] / img.shape[1]
        height = item[1][3] / img.shape[0]
        center_x = tl_x + 0.5*width
        center_y = tl_y + 0.5*height
        left_increase_param = 0.5*width
        right_increase_param = 0.5*width
        up_increase_param = 0.5*height
        down_increase_param = 0.5*height
        Class = 0
        # ##validation
        # while int((center_x-left_increase_param)*img_resized.shape[1]) < 0:
        #     left_increase_param = 0.95*left_increase_param
        # while int((center_x-left_increase_param)*img_resized.shape[1]) > 416:
        #     left_increase_param = 1.05*left_increase_param
        
        # while int((center_x+right_increase_param)*img_resized.shape[1]) < 0:
        #     right_increase_param = 0.95*right_increase_param
        # while int((center_x+right_increase_param)*img_resized.shape[1]) > 416:
        #     right_increase_param = 1.05*right_increase_param

        # while int((center_y+up_increase_param)*img_resized.shape[1]) < 0:
        #     up_increase_param = 1.05*up_increase_param
        # while int((center_y+up_increase_param)*img_resized.shape[1]) > 416:
        #     up_increase_param = 0.95*up_increase_param

        # while int((center_y-down_increase_param)*img_resized.shape[1]) < 0:
        #     down_increase_param = 1.05*down_increase_param
        # while int((center_y-down_increase_param)*img_resized.shape[1]) > 416:
        #     down_increase_param = 0.95*down_increase_param

        pt1_x = int((center_x-left_increase_param)*img_resized.shape[1])
        pt2_x = int((center_x+right_increase_param)*img_resized.shape[1])
        pt1_y = int((center_y+up_increase_param)*img_resized.shape[1])
        pt2_y = int((center_y-down_increase_param)*img_resized.shape[1])

        cv2.rectangle(img_resized, pt1=(pt1_x,pt1_y), pt2=(pt2_x,pt2_y), color=(0,0,255), thickness=3)
        cv2.circle(img_resized, (int(center_x*img_resized.shape[0]),int(center_y*img_resized.shape[1])), radius=0, color=(0, 255, 0), thickness=5)
        cv2.imshow('img_resized',img_resized)
        cv2.waitKey(1)
        center_x_after_modification = ((center_x-left_increase_param)+(center_x+right_increase_param))/2
        center_y_after_modification = ((center_y+up_increase_param)+(center_y-down_increase_param))/2
        width_after_modification = abs((center_x-left_increase_param)-(center_x+right_increase_param))
        height_after_modification = abs((center_y+up_increase_param)-(center_y-down_increase_param))
        with open(Path(output_dir_test.as_posix() + '/' + item[0].name.split('.')[0] + '.txt').as_posix(), 'w') as f:
            f.write(f'{Class} {center_x_after_modification} {center_y_after_modification} {width_after_modification} {height_after_modification}')
        cv2.imwrite(Path(output_dir_test.as_posix() + '/' + item[0].name.split('.')[0] + '.png').as_posix(),img_resized_org)





def train_and_test_txt_file_organizer(train_file_path:Path, test_file_path:Path,train_dir_path:Path, test_dir_path:Path, vgg16:bool=False):
    if vgg16:
        with open(train_file_path.as_posix(), 'a') as f:
            for image_file in list(train_dir_path.glob('*.png')):
                f.write(image_file.as_posix() + '\n')
        f.close()

        with open(test_file_path.as_posix(), 'a') as f:
            for image_file in list(test_dir_path.glob('*.png')):
                f.write(image_file.as_posix() + '\n')
        f.close()
    else:
        train_dir_list = shuffle(list(train_dir_path.glob('*.png')))
        with open(train_file_path.as_posix(), 'a') as f:
            for image_file in train_dir_list:
                f.write(image_file.as_posix() + '\n')
        f.close()

        with open(test_file_path.as_posix(), 'a') as f:
            for image_file in list(test_dir_path.glob('*.png')):
                f.write(image_file.as_posix() + '\n')
        f.close()








if __name__ == '__main__':
    add_augmentations()
    # train_test_split_mine()
    # data_organizer((640,640),Path(r"/run/user/1000/gvfs/smb-share:server=qnap.lsports.eu,share=data-new/Asaf/scoreboard_detection_data/Tennis/Scoreboards/Annotation"),Path(r"/run/user/1000/gvfs/smb-share:server=qnap.lsports.eu,share=data-new/Asaf/scoreboard_detection_data/Tennis/Scoreboards/Images"),Path(r"/run/user/1000/gvfs/smb-share:server=qnap.lsports.eu,share=data-new/Asaf/scoreboard_detection_data/ready_to_go_data_17_11_2022"), Path(r"/run/user/1000/gvfs/smb-share:server=qnap.lsports.eu,share=data-new/Asaf/scoreboard_detection_data/ready_to_go_data_17_11_2022"))
    # data_organizer_supervisely_peter_script_output((640,640),Path(r"/media/access/New Volume/tmp_output/tmp_table_tennis_addition/labels"),Path(r"/media/access/New Volume/tmp_output/tmp_table_tennis_addition/images"),Path(r"/media/access/New Volume/tmp_output/tmp_table_tennis_addition_after_peter_script"))
    # train_and_test_txt_file_organizer(Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/train.txt"), Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/val.txt"),Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/train"), Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/val"))
    # Data_validator((640,640), Path(r"C:\Coding projects\yolo\data\ready_to_go_data"), sample_size=1000)
    # checking_intersection_and_generating_annotations_dict()
    
    a = 2
























