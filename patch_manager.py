import utils
from itertools import chain


class ImagePatchManager:
    def __init__(self, image_list, patch_size):
        """
        image_list -- list of images (np.array)
        patch_site -- size of patch
        """
        self._images = image_list
        self.patch_size = patch_size

        self.ids, self.patches = [], []
        idx = 0
        for img in self._images:
            start = idx
            for patch in utils.img_crop(img, patch_size, patch_size):
                self.patches.append(patch)
                idx += 1
            end = idx
            self.ids.append((start, end))
        assert len(self.ids) == len(self._images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, slice_ob):
        """To support magr[start, stop, end]"""
        start, _= self.ids[slice_ob[0]]
        _, stop = self.ids[slice_ob[1]]

        return self.patches[start : stop + 1]

    def get_p(self, start_image_idx, stop_image_idx=None, map_f=None, flatmap=None):
        """Allows to retrieve patches and to apply functions to them
        start_image_idx -- index of the first image whose patch are extracted (inclusive)
        stop_image_idx -- index of last image whose patch are extracted (exclusive)
        map_f -- (optional) function to be applied to each patch
        flatmap -- (optional) function to apply to each patch which returns an iterable to be flattened
        """
        if stop_image_idx is None:
            stop_image_idx = start_image_idx + 1

        patches = self[start_image_idx, stop_image_idx]

        if map_f:
            patches = map(map_f, patches)
        if flatmap:
            patches = chain.from_iterable(map(flatmap, patches))
        try:
            return list(patches)
        except:
            return [patches]

    def get(self, img_idx):
        """Image getter"""
        return self._images[img_idx]

