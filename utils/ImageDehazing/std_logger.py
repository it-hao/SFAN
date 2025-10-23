import sys


class STD_Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass


if __name__ == "__main__":
    sys.stdout = STD_Logger('log.log', sys.stdout)
    sys.stderr = STD_Logger('log.log', sys.stderr)
    print("fuse ")
    print(a + b)

    # 对logging实例输出的info进行记录
    # handler = logging.FileHandler(os.path.join(project, name, "log.log"))
    # handler.setLevel(level=logging.INFO)
    # LOGGER.addHandler(handler)
