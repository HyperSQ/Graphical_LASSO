# Graphical_LASSO

Packages required:
```
numpy(= 1.26.4) pandas matplotlib scikit-learn tushare h5py gglasso pytz
```

```numpy``` 版本略低，推荐创建虚拟环境

```show_data.py```用于展示```/data/```中已经计算好的GL序列

```SGL(FGL)_MX_WY_PZ(_R).pkl```文件名含义为用SGL（FGL），组合MX，观测窗口长Y，时间轴长度约Z年，（样本外回测数据） 

```GL.py GL_anl``` 用于计算GL 

```get_data.py``` 从Tushare中获取日线数据

```example.py``` 调用计算样例

```app.R``` Rshiny可视化，[Rshiny_GL](https://hypersq.shinyapps.io/Graph_LASSO/)
