#! /usr/bin/python
#
# Implemented by Jiayu Wang (email: jiayuw6@student.unimelb.edu.au). Copyright reserved.
#


class DBHNode:
    def __init__(self, func_index=None, left_child=None, right_child=None, size=1):
        self._func_index = func_index
        self._left_child = left_child
        self._right_child = right_child
        self._size = size

    def get_children(self):
        if self._left_child is None and self._right_child is None:
            return None
        else:
            return [self._left_child, self._right_child]

    def get_left_child(self):
        return self._left_child

    def get_right_child(self):
        return self._right_child

    def get_index(self):
        return self._func_index

    def get_size(self):
        return self._size
