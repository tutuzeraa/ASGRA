'''
Places8 Dataset
'''
import re
import json
import string
import logging

from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)

_punct  = string.punctuation                    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
_stop   = {"a", "an", "the"}                    # determiner prefixes
_re_bad = re.compile(r"(^|\s)mask\]?\s*")       # “mask]…”
_re_and = re.compile(r"\b(and|,|/)\b")          # split tokens

def normalise(raw: str) -> str:
    """ Clean words """
    if not raw:
        return "<unk>"
    t = raw.lower().strip()
    t = _re_bad.sub(" ", t)
    t = t.strip(_punct + "[]{}() ")

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

    return t if t else "<unk>"


class Places8SceneGraphDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("graphs_rank-*part-*.jsonl")) 
        if not self.files:
            raise FileNotFoundError(f"No *.jsonl files in {data_dir}")
        
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.rel2idx = defaultdict(lambda: len(self.rel2idx))
        
        logger.info(f"Loading {len(self.files)} JSONL files - building vocabulary ...")

        self._n_graphs = 0
        for fp in self.files:
            with fp.open() as fh:
                for line in fh: 
                    self._n_graphs += 1
                    self._scan_line(json.loads(line))

        super().__init__(root=None)

        logger.info("Finished ✅. Loaded %d graphs, %d object tokens, %d relations.",
                    self._n_graphs, len(self.word2idx), len(self.rel2idx))
        
        # print("---------------")
        # print(word2idx)
        # print("---------------")
        # print(rel2idx)


    def len(self):
        return self._n_graphs


    def get(self, idx):
        for fp in self.files:
            with fp.open() as fh:
                for line in fh:
                    if idx == 0:
                        return self._scan_line(json.loads(line))
                    idx -= 1
        raise IndexError
    

    def _scan_line(self, js: dict) -> Data:
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
        graph_label = self._idx(self._scene_label(js["image_id"]["image_id"]), self.word2idx)  # or keep str

        # build nodes ------------------------------------------------------------
        node_idx = {}                  
        x_list, box_list = [], []

        for t in js["triplets"]:
            for role in ("sub", "obj"):
                ent = normalise(t[role])
                if ent not in node_idx:
                    node_idx[ent] = len(node_idx)
                    x_list.append(self._idx(ent, self.word2idx))
                    box_list.append(torch.tensor(t[f"{role[:3]}_box"], dtype=torch.float))

        # build edges ------------------------------------------------------------
        src, dst, rel = [], [], []
        for t in js["triplets"]:
            s_id = node_idx[normalise(t["sub"])]
            o_id = node_idx[normalise(t["obj"])]
            src.append(s_id);  dst.append(o_id)
            rel.append(self._idx(normalise(t["pred"]), self.rel2idx))

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
                    y=torch.tensor([graph_label], dtype=torch.long))


    def _scene_label(self, image_id: str) -> str:
        '''
        Get groundtruth from image_id
        '''
        return Path(image_id).parts[1] # 'b/bathroom/00000001.jpg' → 'bathroom'


    def _idx(self, word: str, table: defaultdict) -> int:
        return table.setdefault(normalise(word), len(table))
