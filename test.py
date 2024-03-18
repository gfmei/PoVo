from datasets.data_util import custom_collate_fn
from datasets.matterport import Matterport
from datasets.meta_data.label_constants import MATTERPORT_LABELS_160

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from libs.vis_utils import creat_labeled_point_cloud, get_colored_point_cloud_pca_sep, draw_superpoints

    data_root = '/data/disk1/data/Matterport'
    # data_root = '/storage2/TEV/datasets/Matterport'
    test_data = Matterport(data_root, vis_name='textclip', split='test', img_size=(640, 512))
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    for i, data in enumerate(test_dataloader):
        pcd, feat, normal, llava_cls_names, opcd, llavalb_emd, gt_labelid = data
        # get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), gtlb_emd[0].detach().cpu().numpy(), 'result/gtpca')
        # get_colored_point_cloud_from_labels(pcd[0].detach().cpu().numpy(), gamma[0].detach().cpu().numpy(), name='clus')
        # draw_point_cloud(rpoints[0].cpu().numpy(), None, f'result/org{str(i)}')
        pdlb_embed = llavalb_emd[0]
        # gt_ids = torch.argmax(torch.einsum('md,nd->mn', gtpcd_emd[0],
        #                                    llavalb_emd[0].to(gtpcd_emd[0])), dim=-1).flatten().tolist()
        # creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), gt_ids, f'result/gt{str(i)}')
        # score = torch.einsum('md,nd->mn', test_data.lb_emd, llavalb_emd[0])
        ids2cat = list(MATTERPORT_LABELS_160)
        ids2cat.append('otherfurniture')
        # uni_ids = set(gt_labelid[0].tolist())
        print(set(llava_cls_names[0]).intersection(set(ids2cat)))
        get_colored_point_cloud_pca_sep(pcd[0].detach().cpu().numpy(), feat[0].detach().cpu().numpy(),
                                        f'result/pca{str(i)}')
        draw_superpoints(pcd[0].detach().cpu().numpy(), normal[0].detach().cpu().numpy(), f'result/spts{str(i)}')
        pred_scores = 1 + torch.einsum('md,nd->mn', feat[0], pdlb_embed.to(feat[0]))
        # d_pcd, pred_labels1 = max_vote(pcd[0].to(pred_scores), normal[0].to(pred_scores), pred_scores.to(pred_scores))
        pred_labels = torch.argmax(pred_scores, dim=-1)
        creat_labeled_point_cloud(pcd[0].detach().cpu().numpy(), pred_labels.flatten().tolist(), f'result/pred{str(i)}')
        # Save using pickle
        print('The {}th point cloud has been processed'.format(i))
        with open(f'result/pcd_{i}.pickle', 'wb') as f:
            pickle.dump({'pcd': pcd[0], 'pred_names': [llava_cls_names[0][ids] for ids in pred_labels],
                         # 'spts': [llava_cls_names[0][ids] for ids in pred_labels],
                         'gt_names': [ids2cat[ids] for ids in gt_labelid[0].tolist()]}, f)
        # # print(update_names, ids2cat)
        # print(i, data[0].shape)
        #     # gamma = fusion_wkeans([pcd.float(), normal.float()],
        #     #                       [None, 1], n_clus=20, iters=10, is_prob=False, idx=0)[0]
        #     print(pcd.shape, feat.shape, txt_emd.shape)
        #     # label_np = list(set(labels[0].to(torch.int32).tolist()))
        #     # label_np = set(labels[0].to(torch.int32).tolist())
        #     # color = colors[0].detach().cpu().numpy()
        # draw_point_cloud(org_pcd[0].detach().cpu().numpy(), None, 'orig')
        # draw_point_cloud(superpoints[0].detach().cpu().numpy(), None, 'spts')
        if i > 5:
            break
