import os

import open3d as o3d
import numpy as np
import torch

from datasets.data_util import custom_collate_fn
from datasets.meta_data.scannet200_constants import CLASS_LABELS_200
from datasets.scannet import ScanNet
from libs.o3d_util import generate_mesh_from_pcd, get_super_point_cloud
from models.segmodel import max_vote


def print_hi(knn=33):
    # Path to your PLY file
    ply_path = 'datasets/result/gt10.ply'
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    xyz = np.asarray(pcd.points)
    print(pcd.normals)
    # To visualize the point cloud (optional)
    # o3d.visualization.draw_geometries([pcd])
    mesh = generate_mesh_from_pcd(xyz, normal=None)
    spts = get_super_point_cloud(mesh)
    print(spts)
    # mesh.show()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from libs.vis_utils import creat_labeled_point_cloud, get_colored_point_cloud_pca_sep, draw_point_cloud

    data_root = '/data/disk1/data/scannet'
    # data_root = '/storage2/TEV/datasets/ScanNet'
    test_data = ScanNet(data_root, vis_name='textclip', split='val', img_size=(320, 240), is_orig=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for i, data in enumerate(test_dataloader):
        pcd, feat, normal, llava_cls_names, opcd, llavalb_emd, gt_labelid = data
        pdlb_embed = llavalb_emd[0]
        gt_ids = gt_labelid[0].flatten().tolist()
        creat_labeled_point_cloud(opcd[0].detach().cpu().numpy(), gt_ids, f'scan_result/gt{str(i)}')
        # score = torch.einsum('md,nd->mn', test_data.lb_emd, llavalb_emd[0])
        ids2cat = list(CLASS_LABELS_200)
        ids2cat.append('otherfurniture')
        get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), feat[0].detach().cpu().numpy(),
                                        f'scan_result/pca{str(i)}')
        # pred_labels = torch.argmax(torch.einsum('md,nd->mn', feat[0], pdlb_embed), dim=-1).flatten().tolist()
        pred_scores = 1 + torch.einsum('md,nd->mn', feat[0], pdlb_embed.to(feat[0]))
        pred_label_w = torch.argmax(pred_scores, dim=-1).detach().cpu().numpy()
        print(set(pred_label_w.flatten().tolist()))
        creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), pred_label_w.flatten().tolist(), f'scan_result/predw{str(i)}')
        output_file = os.path.join(f'scan_result', f'name{str(i)}{i}_llava.txt')
        print('Finish one scene ...')
        with open(output_file, "w") as file:
            for item in llava_cls_names[0]:
                file.write(item + "\n")
        # Save using pickle
        print('The {}th point cloud has been processed'.format(i))
        # with open(f'result/pcd_{i}.pickle', 'wb') as f:
        #     pickle.dump({'tensor': pcd[0], 'pred_names': [llava_cls_names[0][ids] for ids in pred_labels],
        #                  'gt_names': [ids2cat[ids] for ids in gt_labelid[0].tolist()]}, f)
        # # print(update_names, ids2cat)
        # print(i, data[0].shape)
        #     # gamma = fusion_wkeans([pcd.float(), normal.float()],
        #     #                       [None, 1], n_clus=20, iters=10, is_prob=False, idx=0)[0]
        #     print(pcd.shape, feat.shape, txt_emd.shape)
        #     # label_np = list(set(labels[0].to(torch.int32).tolist()))
        #     # label_np = set(labels[0].to(torch.int32).tolist())
        #     # color = colors[0].detach().cpu().numpy()
        draw_point_cloud(opcd[0].detach().cpu().numpy(), None, f'scan_result/orig{str(i)}')
        # draw_point_cloud(superpoints[0].detach().cpu().numpy(), None, 'spts')
        if i == 0:
            break

