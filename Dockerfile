# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-py3 

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## Install Requirements（requirements.txt包含python包的版本）
## 这里使用清华镜像加速安装
RUN pip --default-timeout=3600 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


## 镜像启动后统一执行 sh run.sh
USER root
CMD ["sh", "run.sh"]

 