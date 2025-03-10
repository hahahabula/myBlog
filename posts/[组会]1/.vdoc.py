# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Times New Roman, SimSun'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
# 定义函数 f(x, y) = sqrt(x^2 + y^2)
def f(x, y):
    return np.sqrt(x**2 + y**2)
# 生成数据
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
# 创建三维图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10, label='f(x, y)')
# 设置标签
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_zlabel('f(x, y)')
# 设置标题
ax.set_title(r'$f(x, y) = \sqrt{x^2 + y^2}$')
plt.show()
#
#
#
#
#
#
