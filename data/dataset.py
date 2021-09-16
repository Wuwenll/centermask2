import os
import glob
import pickle

import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from shapely.geometry import Polygon
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from pathlib import Path
_PREDEFINED_SPLITS_CHAR = {
    "data00": ("data00/data", "data00/label"),
    "data01": ("data01/data", "data01/label"),
    "data02": ("data02/data", "data02/label"),
    "data03": ("data03/data", "data03/label"),
    "data04": ("data04/data", "data04/label"),
    "data05": ("data05/data", "data05/label"),
    "data06": ("data06/data", "data06/label"),
    "data07": ("data07/data", "data07/label"),
    "data08": ("data08/data", "data08/label"),
    "data09": ("data09/data", "data09/label"),
    "data10": ("data10/data", "data10/label"),
    "data11": ("data11/data", "data11/label"),
    "data12": ("data12/data", "data12/label"),
    "data13": ("data13/data", "data13/label"),
    "data14": ("data14/data", "data14/label"),
    "data15": ("data15/data", "data15/label")
}

TRUE_DATASET = [
    "data03", "data04", "data05", "data06", "data07", "data14"
]

_PREDEFINED_TEST_CHAR = {
    "unreal_test": ("UnrealText/test", "UnrealText/test/test.json"),
    "P1868_test": ("viditestData/P1868/test/images", "viditestData/P1868/instances_test.json"),
    "P1871_test": ("viditestData/P1871/test/images", "viditestData/P1871/instances_test.json"),
    "P1879_test": ("viditestData/P1879/test/images", "viditestData/P1879/instances_test.json"),
    "P1881_test": ("viditestData/P1881/test/images", "viditestData/P1881/instances_test.json"),
    "P1885_test": ("viditestData/P1885/test/images", "viditestData/P1885/instances_test.json"),
    "P1897_test": ("viditestData/P1897/test/images", "viditestData/P1897/instances_test.json"),
    "P2217_test": ("viditestData/P2217/test/images", "viditestData/P2217/instances_test.json"),
    "P2218_test": ("viditestData/P2218/test/images", "viditestData/P2218/instances_test.json"),
    "P2220_test": ("viditestData/P2220/test/images", "viditestData/P2220/instances_test.json"),
    "P2222_test": ("viditestData/P2222/test/images", "viditestData/P2222/instances_test.json"),
    "P2224_test": ("viditestData/P2224/test/images", "viditestData/P2224/instances_test.json"),
    "P2225_test": ("viditestData/P2225/test/images", "viditestData/P2225/instances_test.json"),
    "P2226_test": ("viditestData/P2226/test/images", "viditestData/P2226/instances_test.json"),
    "P2227_test": ("viditestData/P2227/test/images", "viditestData/P2227/instances_test.json"),
    "P2228_test": ("viditestData/P2228/test/images", "viditestData/P2228/instances_test.json"),
}

metadata_CHAR = {
    "thing_classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
}

metadata_CHAR94 = {
    "thing_classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', 
    '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', 
    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
}

metadata_CHAR36 = {
    "thing_classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
}


char2id = {c: i for i, c in enumerate(metadata_CHAR["thing_classes"])}
categories = [{'id': v, 'name': k} for k, v in char2id.items()]


def get_image(img_path):
    assert os.path.exists(img_path), f"{img_path} not found"
    img = Image.open(img_path)
    img_w, img_h = img.size[0], img.size[1]
    del img
    return img_path, img_h, img_w


def xyxy2xywh(bbox, max_w, max_h):
    bbox = np.array(bbox, np.int32).reshape(2, 2)
    bbox[:, 0] = bbox[:, 0].clip(0, max_w)
    bbox[:, 1] = bbox[:, 1].clip(0, max_h)
    bbox[1] = bbox[1] - bbox[0]
    return bbox.reshape(-1).tolist()


def register_vimo_format_dataset(root="datasets"):
    null_dataset = []
    for key, (image_dir, label_dir) in _PREDEFINED_SPLITS_CHAR.items():
        # Assume pre-defined datasets live in `./datasets`.
        image_dir = os.path.join(root, image_dir)
        label_dir = os.path.join(root, label_dir)
        image_paths = glob.glob(os.path.join(image_dir, "*"))[:1000000]
        if not len(image_paths) > 0:
            null_dataset.append(key)
            continue
        register_vimo_format_instances(key, image_dir, label_dir, image_paths)
    assert len(null_dataset) != len(_PREDEFINED_SPLITS_CHAR), "All dataset is NULL!"


def register_vimo_format_instances(name, image_dir, label_dir, image_paths):
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_vimo_format_data(name, image_dir, label_dir, image_paths))
    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_dir=image_dir, label_dir=label_dir, filenames=image_paths, evaluator_type="coco", **metadata_CHAR
    )


def zoom_the_annotation(points, ratio=-0.1):
    xs = points[:, 0]
    ys = points[:, 1]

    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    new_xs = np.array([(i - x_center) * (1 - ratio) + x_center for i in xs])
    new_ys = np.array([(i - y_center) * (1 - ratio) + y_center for i in ys])

    points[:, 0] = new_xs
    points[:, 1] = new_ys
    points = np.array(points).reshape(-1, 2).astype(np.int32)
    return points


def load_vimo_format_data(data_name, image_dir, label_dir, image_paths):

    cache_path = f"datasets/data_cache/{len(char2id)}_{data_name}.pk"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as a:
            dataset_dicts = pickle.load(a)
            a.close()
        return dataset_dicts
    dataset_dicts = []
    img_idx = 0
    with Pool(processes=1) as pool:
        with tqdm(total=len(image_paths), desc='Scanning images') as pbar:
            for img_path, img_h, img_w in pool.imap_unordered(get_image, image_paths):
                record = {}
                ann_path = img_path.replace(image_dir, label_dir) + ".json"
                assert os.path.exists(img_path) and os.path.exists(ann_path), f"{img_path} or {ann_path} not found"
                record["file_name"] = img_path
                record["height"] = img_h
                record["width"] = img_w
                record["image_id"] = img_idx
                record["data_name"] = data_name
                objs = []
                try:
                    with open(ann_path, "r") as fp:
                        ann = json.load(fp, encoding='utf-8')
                        ann_labels = ann["Labels"]
                        for lb in ann_labels:
                            char = lb["Comment"]
                            if char not in char2id:
                                continue
                            points = []
                            for p in lb["Points"]:
                                points.append([p['X'], p['Y']])
                            points = np.array(points, dtype=int).reshape(-1, 2)
                            if data_name in TRUE_DATASET:
                                points = zoom_the_annotation(points)
                            poly = Polygon(points)
                            bbox = np.array(poly.bounds)
                            objs.append({
                                'iscrowd': 0,
                                'bbox': xyxy2xywh(bbox, img_w, img_h),
                                # 'category_id': char2id.get(char, char2id["#"]),
                                'category_id': char2id.get(char),
                                'segmentation': [points.reshape(-1).tolist()],
                                'bbox_mode': BoxMode.XYWH_ABS
                            })
                    if len(objs):
                        record["annotations"] = objs
                        dataset_dicts.append(record)
                        img_idx += 1
                except Exception as e:
                    print(e)
                    print(ann_path)
                    raise e
                pbar.update()
            pool.close()
    assert len(dataset_dicts) > 0
    os.makedirs(Path(cache_path).parent, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset_dicts, f)
        f.close()
    return dataset_dicts



