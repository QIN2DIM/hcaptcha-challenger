## Introduction

内存诊断，评估可插拔模型在并行程序中的性能释放效率

## Quick start

安装 [内存探针](https://github.com/pythonprofilers/memory_profiler) 和作图工具

```bash
pip install -U memory-profiler matplotlib
```

安装完后需要在关键执行点做探针埋点，为函数投掷一个修饰器 `@profile` 即可，不需要 import 依赖，忽略“错误提示”。

运行脚本，采集多进程内存占用数据。在 demo 中使用不同的 sitekey 以模拟 成功/失败 等多种复制的情况

```bash
mprof run examples/demo_undetected_playwright.py
```

画图

```bash
mprof plot --output benchmarks/mprof-plot.png -s
mprof plot --output benchmarks/mprof-plot-flame.png -f 
```