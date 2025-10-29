'''
Places8 Dataset
'''
import os
import re
import json
import csv
import string
import logging

from difflib import get_close_matches
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)

UNK_TOKEN = "[UNK]"
VG150_OBJECTS = {"[UNK_OBJ]": 0, "kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103, "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11, "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59, "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36, "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112, "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124, "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30, "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116, "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117, "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101, "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64, "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5, "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47, "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26, "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50, "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121, "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7, "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58, "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22, "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44, "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}
VG150_RELS = {"[UNK_REL]": 0, "and": 5, "says": 39, "belonging to": 9, "over": 33, "parked on": 35, "growing on": 18, "standing on": 41, "made of": 27, "attached to": 7, "at": 6, "in": 22, "hanging from": 19, "wears": 49, "in front of": 23, "from": 17, "for": 16, "watching": 47, "lying on": 26, "to": 42, "behind": 8, "flying in": 15, "looking at": 25, "on back of": 32, "holding": 21, "between": 10, "laying on": 24, "riding": 38, "has": 20, "across": 2, "wearing": 48, "walking on": 46, "eating": 14, "above": 1, "part of": 36, "walking in": 45, "sitting on": 40, "under": 43, "covered in": 12, "carrying": 11, "using": 44, "along": 4, "with": 50, "on": 31, "covering": 13, "of": 30, "against": 3, "playing": 37, "near": 29, "painted on": 34, "mounted on": 28}
SCENES = {"bathroom":0, "bedroom":1, "childs_room": 2, "classroom": 3, "dressing_room": 4, "living_room": 5, "studio": 6, "swimming_pool": 7}

VG150_objs_list = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
VG150_rels_list = ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]
ALL_CANON = sorted(VG150_objs_list + VG150_rels_list)

_punct  = string.punctuation                    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
_stop   = {"a", "an", "the"}                    # determiner prefixes
_re_bad = re.compile(r"(^|\s)mask\]?\s*")       # “mask]…”
_re_and = re.compile(r"\b(and|,|/)\b")          # split tokens

def _closest_canon(word: str, cutoff=0.8) -> str | None:
    """
    Return the closest canonical VG150 token if similarity ≥ cutoff,
    else None.  difflib uses a quick ratio (≈ Levenshtein).
    """
    matches = get_close_matches(word, ALL_CANON, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def normalise(raw: str) -> str:
    """ Clean words """
    if not raw:
        return "<unk>"
    
    if raw in ALL_CANON: return raw

    t = raw.lower().strip()
    t = _re_bad.sub(" ", t)                    # remove 'mask]' artefact
    t = t.strip(_punct + " []{}()")
    t = re.sub(r"\b(a|an|the)\s+", "", t)      # drop determiners
    t = re.sub(r"\band\s+", "", t)             # drop leading "and "
    t = re.sub(r"\s+", " ", t).strip()

    words = t.split()
    if words and words[0] in _stop:
        words = words[1:]
    t = " ".join(words)

    parts = _re_and.split(t)
    for p in parts:
        p = p.strip(_punct + " ")
        if p:
            t = p
            break

    t = re.sub(r"\s+", " ", t).strip(_punct + " ").strip()

    if t not in ALL_CANON:
        best = _closest_canon(t)
        if best:
            t = best
        else:
            t = None

    return t if t else None

def _graph_key(fp: str) -> str:
    return os.path.basename(fp)


class Places8SceneGraphDataset(Dataset):
    def __init__(self, data_dir, csv_file, thr = 0, split = None, cache_file = None):
        """
        data_dir : directory with the graphs
        csv_file : places8 split csv
        split    : 'train' | 'val' | 'test' 
        """
        self.word2idx  = VG150_OBJECTS
        self.rel2idx   = VG150_RELS
        self.scene2idx = SCENES

        self.split = split.lower()

        self.score_thr = thr

        self.cache_file = (Path(cache_file).with_name(f"{Path(cache_file).stem}_{split.lower()}.pt")) if cache_file else None
        print(self.cache_file)

        if self.cache_file and self.cache_file.is_file():
            self.data = torch.load(self.cache_file)
            logger.info(f"[cached] {split} loaded from {self.cache_file}, a total of {len(self.data)} graphs loaded.")
            super().__init__(root=None)
            return

        self.files = sorted(Path(data_dir).glob("graphs_rank-*part-*.jsonl")) 
        if not self.files:
            raise FileNotFoundError(f"No *.jsonl files in {data_dir}")
        
        self._img2meta = {}

        with open(csv_file, newline="") as fh:
            reader = csv.reader(fh)
            first_row = next(reader)
            has_header = first_row[0].strip().lower() in {"id", "idx", "index"}
            if not has_header:
                _, fp, label, spl = first_row
                self._img2meta[_graph_key(fp)] = (label, spl.lower())
            for row in reader:
                _, fp, label, spl = row
                self._img2meta[_graph_key(fp)] = (label, spl.lower())

        
        # self.word2idx = defaultdict(lambda: len(self.word2idx))
        # self.rel2idx = defaultdict(lambda: len(self.rel2idx))
        # self.scene2idx = defaultdict(lambda: len(self.scene2idx))
        
        logger.info(f"Loading {len(self.files)} JSONL files - building vocabulary ...")

        # key = b/bedroom/00018858.jpg
        # print(self._img2meta)

        self.data = []
        for fp in self.files:
            with fp.open() as fh:
                for line in fh: 
                    js = json.loads(line)
                    key = _graph_key(js["image_id"]["image_id"])
                    meta = self._img2meta.get(key)
                    if not meta:
                        continue
                    scene_label, spl = meta
                    if spl != self.split:
                        continue
                    self.data.append(self._scan_line(js, scene_label, js["image_id"]["image_id"]))

        super().__init__(root=None)

        logger.info("Finished ✅. For split %s: Loaded %d graphs, %d object tokens, %d relations, and %d classes",
                    self.split, len(self.data), len(self.word2idx), len(self.rel2idx), len(self.scene2idx))
        
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.data, self.cache_file)
            logger.info(f"[cached] saved {split} to {self.cache_file}")

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def _idx(self, word: str, table: dict) -> int:
        return table[word]
    
    def _canon_obj(self, word: str) -> str:
        w = normalise(word)
        # print(f"started with {word}, word is now {w}")
        return w if w in VG150_OBJECTS else "[UNK_OBJ]"

    def _canon_rel(self, word: str) -> str:
        w = normalise(word)
        return w if w in VG150_RELS else "[UNK_REL]"
    

    def _scan_line(self, js: dict, scene_label: str, img_key: str) -> Data:
        """
        Transforms each line of the jsonl into an Data object (Graph for PyG)
        js = {
        "image_id": "b/bathroom/00000001.jpg",
        "triplets": [
            {"subject": "dog", "object": ". door", "predicate": "behind",
            "sub_box": [x1,y1,x2,y2], "obj_box": [...]},
            ...
        ]
        }
        """
        graph_label = self._idx(scene_label, self.scene2idx) 

        # build nodes ------------------------------------------------------------
        node_idx = {}                  
        x_list, box_list = [], []

        kept = [t for t in js["triplets"] if t.get("score", 1.0) >= self.score_thr]
        if not kept:
            return None

        for t in kept:
            for role in ("sub", "obj"):
                ent = self._canon_obj(t[role])
                if ent not in node_idx:
                    node_idx[ent] = len(node_idx)
                    x_list.append(self._idx(ent, self.word2idx))
                    box_list.append(torch.tensor(t[f"{role[:3]}_box"][:4], dtype=torch.float))

        # build edges ------------------------------------------------------------
        src, dst, rel = [], [], []
        for t in kept:
            s_id = node_idx[self._canon_obj(t["sub"])]
            o_id = node_idx[self._canon_obj(t["obj"])]
            src.append(s_id);  dst.append(o_id)
            rel.append(self._idx(self._canon_rel(t["pred"]), self.rel2idx))
            # print(f"obj is {obj}, sub is {sub}, rel is {pred}")

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(rel, dtype=torch.long)

        # node features: [token_id, bbox]  
        x = torch.stack([
            torch.cat([torch.tensor([tok], dtype=torch.long), box])   # 1 + 4 dims
            for tok, box in zip(x_list, box_list)
        ])

        return Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([graph_label], dtype=torch.long),
                    image_id = img_key)
