
import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        
        # ==============================================================
        # 【修改开始】: 为 De-ReID 预训练筛选前 900 个 ID
        # ==============================================================
        # 获取当前加载的所有 PID (train列表里 item[1] 是 pid)
        all_pids = sorted(list(set([item[1] for item in train])))
        
        # 如果 ID 数量超过 900, 则进行截断
        if len(all_pids) > 900:
            target_pids = set(all_pids[:900])  # 取前 900 个 ID
            
            # 重新过滤 train 列表，只保留这 900 人的图片
            train = [item for item in train if item[1] in target_pids]
            
            if verbose:
                print(f"\n[MSMT17 De-ReID] Filter Applied: Keeping first 900 identities.")
                print(f"[MSMT17 De-ReID] Original IDs: {len(all_pids)} -> New IDs: {len(target_pids)}")
                print(f"[MSMT17 De-ReID] Total Train Images: {len(train)}\n")
        # ==============================================================
        # 【修改结束】
        # ==============================================================
          
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path,  self.pid_begin +pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset