from pathlib import Path
import shutil
from tqdm import tqdm

output_path_images = Path(r"C:\Coding projects\yolo\data\ready_to_go_data\images")
output_path_labels = Path(r"C:\Coding projects\yolo\data\ready_to_go_data\labels")

for file in tqdm(Path(r"C:\Coding projects\yolo\data\ready_to_go_data\asafs_old_data").iterdir(), total=len(list(Path(r"C:\Coding projects\yolo\data\ready_to_go_data\asafs_old_data").iterdir()))):
    src = file.as_posix()
    if file.name.split('.')[1] == 'txt':
        dst = output_path_labels.as_posix() + '/' + file.name
    else:
        dst = output_path_images.as_posix() + '/' + file.name
    shutil.copyfile(src, dst)