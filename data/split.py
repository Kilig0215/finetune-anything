import json
import numpy as np


def split_dataset(json_file, ratios, names):
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    assert len(ratios) == len(names), "Must provide name for each split"

    # 载入整个json数据集
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    # 对图片和注解的id进行分割
    image_ids = [image["id"] for image in data["images"]]
    np.random.shuffle(image_ids)
    num_images = len(image_ids)

    splits = [int(ratio * num_images) for ratio in ratios]
    splits[-1] = num_images - sum(splits[:-1])  # Ensure the splits sum to num_images
    split_ids = np.split(image_ids, np.cumsum(splits[:-1]))

    # 创建一个函数来生成新的json数据集
    def create_subset(ids, name):
        subset = {}
        subset["categories"] = data["categories"]
        subset["images"] = [image for image in data["images"] if image["id"] in ids]
        subset["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] in ids]

        # 保存为新的json文件
        with open(f"{name}.json", "w") as write_file:
            json.dump(subset, write_file)

    # 创建数据集
    for ids, name in zip(split_ids, names):
        create_subset(ids, name)


# Example usage:
if __name__=='__main__':
    split_dataset("./convert/coco/coco.json", [0.8, 0.2, 0.0], ["train", "val", "test"])
