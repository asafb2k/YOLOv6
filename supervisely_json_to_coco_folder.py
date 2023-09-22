import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2




def plot_json(png, bbox):
    figure, ax = plt.subplots()
    image = cv2.imread(png.as_posix())
    tl_x, tl_y, br_x, br_y = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    image = cv2.rectangle(img=image,
                          pt1=(tl_x, tl_y),
                          pt2=(br_x, br_y),
                          thickness=2,
                          color=(0, 255, 0),
                          )
    ax.imshow(image)
    plt.show()

if __name__ == '__main__':
    for data_dir in Path(r"C:\Coding projects\yolo\data\data").iterdir():
        
        FOLDER_IN = data_dir.as_posix()
        FOLDER_OUT = Path(rf"C:\Coding projects\yolo\data\tmp_in_coco_format\{data_dir.name}").as_posix()
        LABELS_DICT = {'scoreboard':0}

        path_in = Path(FOLDER_IN)
        path_out_root = Path(FOLDER_OUT)
        path_out_images = path_out_root/'images'/'train'
        path_out_images_val = path_out_root/'images'/'val'
        path_out_labels = path_out_root/'labels'/'train'
        path_out_labels_val = path_out_root/'labels'/'val'

        if path_out_root.exists():
            shutil.rmtree(path_out_root)
        path_out_images.mkdir(parents=True, exist_ok=True)
        path_out_images_val.mkdir(parents=True, exist_ok=True)
        path_out_labels.mkdir(parents=True, exist_ok=True)
        path_out_labels_val.mkdir(parents=True, exist_ok=True)




        pngs = list(path_in.rglob('*.png'))


        yaml_dict = {'colors': [[28, 141, 151], [247, 51, 8], [241, 217, 12], [139, 87, 42], [19, 205, 193]],
                    'names': ['trajectory', 'net', 'bounce', 'Serve Near', 'Serve Far'],
                    'nc': 5,
                    'train': f'../{path_out_images.as_posix()}',
                    'val': f'../{path_out_images_val.as_posix()}'}


        with open(path_out_root/'data_config.yaml', 'w') as f2:
            docs = yaml.safe_dump(yaml_dict, f2,default_flow_style=None)

        skipped_count=0
        for png in tqdm(pngs, total=len(pngs)):
            bsave_this_image_and_annotations = False
            new_png_path = path_out_images/png.name
            label_path = png.as_posix().replace('img', 'ann') + '.json'
            new_label_path = f'{path_out_labels/png.stem}.txt'
            with open(label_path, 'r') as f:
                data = json.load(f)
            skipped_flag = 0
            if len(data['tags']) > 0:
                for el in data['tags']:
                    if 'skipped' == el['name']:
                        skipped_count += 1
                        skipped_flag = 1
                        continue
            if skipped_flag == 1:
                print(f'{png.as_posix()} is skipped')
                skipped_flag = 0
                continue
            if len([it for it in data['objects'] if it['classTitle'] == 'scoreboard']) == 1:
                im_height, im_width = np.array(Image.open(png)).shape[:2]
                bbox = [it for it in data['objects'] if it['classTitle'] == 'scoreboard'][0]['points']['exterior']
                width = (bbox[1][0] - bbox[0][0])/im_width
                height = (bbox[1][1] - bbox[0][1])/im_height
                x_center = bbox[0][0]/im_width + width/2
                y_center = bbox[0][1]/im_height  + height/2
                label = [it for it in data['objects'] if it['classTitle'] == 'scoreboard'][0]['classTitle']
                label_id = LABELS_DICT[label]
                string =f'{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
                bsave_this_image_and_annotations = True

            if bsave_this_image_and_annotations:
                with open(new_label_path,'w') as f1:
                    f1.writelines(string)
                shutil.copy(png, new_png_path)
        print(f'skipped count: {skipped_count}')


