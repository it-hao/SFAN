import os, sys

def modify_path(__file__):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 将根目录添加到path中
    sys.path.insert(0, BASE_DIR)
    # 仅仅是简单的append可能还不够，由于优先级的存在，如果append无效，那不仅要添加项目根目录，
    # 还要放到搜索路径的第一位，所以需要sys.path.insert(0,project_root)这样插入到开头
