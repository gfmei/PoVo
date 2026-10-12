"""Ground-truth instance records used by the ScanNet evaluator."""

import numpy as np


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id, coords=None, encode=1000):
        self.encode = encode
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

        if coords is None:
            self.box = np.zeros(6)
        else:
            inst_coords = coords[mesh_vert_instances == instance_id]
            box_min = inst_coords.min(0)
            box_max = inst_coords.max(0)
            self.box = np.concatenate([box_min, box_max])

    def get_label_id(self, instance_id):
        return int(instance_id // self.encode)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()


    def to_dict(self):
        dict = {"instance_id": self.instance_id, "label_id": self.label_id, "vert_count": self.vert_count,
                "med_dist": self.med_dist, "dist_conf": self.dist_conf, "box": self.box}
        return dict


def get_instances(ids, class_ids, class_labels, id2label, coords=None, dataset = 'scannet200'):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id, coords=coords, encode=1000)

        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances
