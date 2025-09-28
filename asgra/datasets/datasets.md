# DATASETS

We evaluate our model in the following datasets:

- **Places8**
- **RCPD**

The instructions to prepare the dataset are bellow:

## Places8

Places8 is a subset of the Places365 where classes are selected to highlight environments most common in Child Sexual Abuse Imagery (CSAI). To utilize it, you first must download the Places365 dataset from the [official website](http://places2.csail.mit.edu/download.html). Then, download the Places8 split [here](https://zenodo.org/records/13910526).

First, you must generate the scene graphs from the Places8 split. This can be done following the instructions in our [Pix2Grp fork](https://github.com/tutuzeraa/Pix2Grp_CVPR2024/tree/a8e9fbb4c4c798c0dd456d1570ff1a524c004a50?tab=readme-ov-file#instructions). After that, move all the generated graphs (jsonl files) to `data/graphs`.

Note that each line (that represents one image), must be formatted like this:

```json
{"image_id": "b/bathroom/00000001.jpg",
"triplets": [
    {"subject": "dog", "object": ". door", "predicate": "behind",
    "sub_box": [x1,y1,x2,y2], "obj_box": [...]},
    ...
]
}
```

## Region-Based Annotated Child Pornography Dataset (RCPD)

The region-based annotated child pornography dataset (RCPD) is a private database that belongs to the Brazilian Federal Police. In the current moment, is not possible to get the graphs from the dataset, but soon we expect to release the graphs. You can get more information at the [official site](https://patreo.dcc.ufmg.br/datasets/rcpd/).