import torch
class YOLOV8AnchorGenerator(object):

    def __init__(self, grid_cell_offset=0.5):
        super(YOLOV8AnchorGenerator, self).__init__()
        self.grid_cell_offset = grid_cell_offset

    def grid_anchors(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(feat_w)
        shift_y = torch.arange(feat_h)
        shift_yy, shift_xx = torch.meshgrid([shift_y, shift_x])
        shifts = torch.stack((shift_xx, shift_yy), dim=-1).view(
            1, feat_h, feat_w, 2)

        anchors = torch.cat([
            shifts + self.grid_cell_offset, shifts + self.grid_cell_offset
        ], dim=-1)
        anchors *= stride
        return anchors

import collections

class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, PIPELINES)
                self.transforms.append(t)
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string