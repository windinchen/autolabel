"""人体骨骼自动标注系统：YOLO-pose 批量标注 + 人工复核 + 训练数据导出。

版本号与开发者信息只在 :mod:`autolabel.version` 里定义，这里转出给
``--version``、界面、接口共用，避免出现第二个要记得改的地方。
"""

from .version import DEVELOPER, REPO_URL, VERSION as __version__

__all__ = ["DEVELOPER", "REPO_URL", "__version__"]
