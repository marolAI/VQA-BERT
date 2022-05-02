import os
import json


class Database:
    def __init__(self, dataroot, split):
        self.dataroot = dataroot
        self.split = split

    def load(self, name):
        if name == "annotations":
            annotations_path = os.path.join(
                self.dataroot,
                "original",
                "Annotations",
                "v2_mscoco_%s_annotations.json",
            )
            return self._load_json(annotations_path)["annotations"]
        elif name == "questions":
            questions_path = os.path.join(
                self.dataroot,
                "original",
                "Questions",
                "v2_OpenEnded_mscoco_%s_questions.json",
            )
            return self._load_json(questions_path)["questions"]
        else:
            print("%s is unknown!" % name)

    def _load_json(self, path):
        if self.split in ["train2014", "val2014"]:
            with open(path % self.split) as f:
                data = json.load(f)
        else:
            print("%s is not a valid split!" % self.split)
        return data


# if __name__ == '__main__':
#     dataroot = '/media/maro/9cbd9ba9-da6e-4d06-baf2-a794e6249205/projects/VQA-BERT/code'


# train2014 = Database(dataroot, 'train2014')
# train_annotations = train2014.load('annotations')
# train_questions = train2014.load('questions')

# val2014 = Database(dataroot, 'val2014')
# val_annotations = val2014.load('annotations')
# val_questions = val2014.load('questions')
