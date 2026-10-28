# Python wheel 兼容索引

## 地址

- 主仓库：`ririv/py-extra-wheels`
- 主包索引：`https://ririv.github.io/py-extra-wheels/`
- 兼容包索引：`https://ririv.github.io/android-wheels/`

兼容索引由本仓库的 `android-wheels/` 目录提供，供使用旧地址的 pip 配置继续安装包。该目录仅存放索引 HTML 和同步版本标记；wheel 二进制统一保存在主仓库的 `wheels` 分支。

## 自动同步

`sync-wheel-index.yml` 工作流约每 15 分钟拉取主仓库的 `gh-pages` 分支，也可手动触发。GitHub 的定时任务可能延迟，兼容索引不保证实时更新。

同步脚本先检查包目录、wheel 链接及源数据完整性，然后只替换 `android-wheels/` 子目录。现有首页、主题和其他站点文件保持原样。上游不完整或链接未指向指定仓库时工作流失败，不发布异常索引。

有变化时，工作流提交兼容索引，再显式调用 Pages 构建接口。工作流只使用本仓库自带的 `GITHUB_TOKEN`，权限限于本仓库的内容写入和 Pages 构建，不需要跨仓库密钥。

即使索引没有变化，工作流仍检查当前提交的发布状态，确保先前中断在提交与发布之间的运行可以继续发布。已成功发布或正在发布的同一提交不会重复触发构建。

`android-wheels/sync-revision.txt` 记录已发布索引对应的上游提交，用于核对同步状态。
