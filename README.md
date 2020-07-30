# PyML-Course
This is a repository for Python Machine Learning Courseware

## Update Python Package

经常更新 Python 包，就有一次需要更新很多个包的情况，目前没有一键升级功能

pip-review 就是为了解决这个问题的第三方包，安装方法如下

```shell
pip install pip-review
```

查看可用更新

```shell
pip-review
```

自动批量升级

```shell
pip-review --auto
```

以交互方式运行，对每个包进行升级

```shell
pip-review --interactive
```
