# Waymo

作用:
    1、解析 waymo 的 tfrecord 文件
    2、将 annotations 和点云数据保存到 annos & lidar 文件下（pickle格式）
    3、将 gt-base 及其他信息保存到一个文件里
    4、evaluation 等

* 注: 代码未独立测试，仅供参考，如需单独运行，需要额外安装 det3d 框架或者进行其他修改


```
waymo_common.py................... from det3d
waymo_converter.py................ from det3d
waymo_decoder.py.................. from det3d

waymo_utils.py.................... from openpcdet
waymo_eval.py..................... from openpcdet
```





