import os
from ctypes import *

dirname = os.path.dirname(__file__)
event_lib = CDLL(os.path.join(dirname, "event_reader.so"))


class Event(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("polarity", c_bool),
        ("timestamp", c_long),
    ]


event_lib.read_window.argtypes = (
    c_char_p,
    POINTER(c_long),
    POINTER(c_long),
    POINTER(Event),
    c_int,
)

event_lib.read_window.restype = c_void_p


# void read_window(int *read_from, long *time_high, Event *event_buffer, int event_buffer_size)


def c_fill_event_buffer(fpath, buffer_size, last_read_from, last_time_high):
    buffer = (Event * buffer_size)(*[])
    filename = fpath.encode("utf-8")
    read_from = c_long(last_read_from)
    time_high = c_long(last_time_high)
    event_lib.read_window(filename, byref(read_from), byref(time_high), buffer, buffer_size)

    return buffer, read_from.value, time_high.value
