#加入#!/usr/bin/python，说明这个是python脚本
#!/usr/bin/python
#加入这句文件内有中文执行不会出错
# -*- coding: UTF-8 -*-

#chmod +x 文件名 给文件添加可执行权限

import os
#不导入io 运行脚本时会报encoding错误
import io
from langconv import *

def simple2Traditional(str):
    str = Converter('zh-hant').convert(str)
    return str

# folder = '../string/'

newPath = os.path.dirname(os.path.realpath(__file__)) + "/convert/"

def file2Traditional(file):
    with io.open(folder + file, 'r', encoding='utf-8') as f:
        s = simple2Traditional(f.read())
        # print(s)
        newFile = io.open(newPath + file, 'w', encoding='utf-8')
        newFile.write(s)
        newFile.close()


if __name__=="__main__":
    folder = sys.argv[1]
    # folder = '../string/'

    fileList = os.listdir(folder)
    for file in fileList:
        print(folder + file)

        isExists = os.path.exists(newPath)
        if not isExists:
            os.makedirs(newPath)
            print(newPath+"创建成功")
        else:
            print("exits")

        file2Traditional(file)

print("文件转化成功->"+newPath)