import os 
import wget
import zipfile


project_dir = '/media/maro/9cbd9ba9-da6e-4d06-baf2-a794e6249205/projects/VQA-BERT/code'
dataroot_dir = os.path.join(project_dir, 'data')

# global dataroot_dir
annotations_dir = os.path.join(dataroot_dir, 'Annotations')
questions_dir = os.path.join(dataroot_dir, 'Questions')

# create the dataset directories if don't yet exist
os.makedirs(dataroot_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(questions_dir, exist_ok=True)


if not os.listdir(annotations_dir):
    print("[INFO] Download and unzip the Annotations files if don't yet exist.")
    split = ['Train', 'Val']
    for s in split:
        annotations_file = annotations_dir + '/' + 'v2_Annotations_%s_mscoco.zip' %s 
        annotations_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_%s_mscoco.zip' %s
        if not os.path.exists(annotations_file):
            wget.download(annotations_url, annotations_file)
        with zipfile.ZipFile(annotations_file, "r") as zip_ref:
            zip_ref.extractall(annotations_dir)
        os.remove(annotations_file)
else:
    print('[INFO] Annotations already downloaded!')

if not os.listdir(questions_dir):
    print("[INFO] Download and unzip the questions files if don't yet exist.")
    split = ['Train', 'Val', 'Test']
    for s in split:
        questions_file = questions_dir + '/' + 'v2_Questions_%s_mscoco.zip' %s 
        questions_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_%s_mscoco.zip' %s
        if not os.path.exists(questions_file):
            wget.download(questions_url, questions_file)
        with zipfile.ZipFile(questions_file, "r") as zip_ref:
            zip_ref.extractall(questions_dir)
        os.remove(questions_file)
else:
    print('[INFO] Questions already downloaded!')
print('[INFO] Done!')