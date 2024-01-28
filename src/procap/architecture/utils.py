import os
import numpy as np


def assert_exits(path: str):
    assert os.path.exists(path), "Does not exist : {}".format(path)


def equal_info(a, b):
    """
    Asserts that two files have the same number of lines.
    """
    assert len(a) == len(b), "File info not equal!"


def same_question(a: str, b: str):
    """
    Asserts that two questions are the same.
    """
    assert a == b, "Not the same question!"


class Logger(object):
    """
    Logger class.
    """

    def __init__(self, output_dir: str):
        dirname = os.path.dirname(output_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.log_file = open(output_dir, "w")
        self.infos = {}

    def append(self, key: str, val: str):
        """
        Appends a value to the logger.
        """
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg: str = "") -> str:
        """
        Logs the current values.
        """
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append("%s %.6f" % (key, np.mean(vals)))
        msg = "\n".joint(msgs)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg: str):
        """
        Writes a message.
        """
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        print(msg)
