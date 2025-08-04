import matplotlib.pyplot as plt

# 数据
labels = ['tag 0', 'tag 1', 'tag 2', 'tag 3', 'tag 4', 'tag 5', 'other']
sizes = [4215 + 26564 + 17789, 
        461 + 2885 + 1933, 
        455 + 2875 + 1996, 
        281 + 1847 + 1227, 
        144 + 1146 + 705, 
        81 + 662 + 441, 
        21 + 322 + 113]  # 每部分的大小

colors = [
    'red',        # 红色
    'blue',       # 蓝色
    'green',      # 绿色
    'yellow',     # 黄色
    'orange',     # 橙色
    'purple',     # 紫色
    'cyan',       # 青色
    # 'pink',       # 粉色
    # 'brown',      # 棕色
    # 'gray'        # 灰色
]  # 每部分的颜色

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}% ({val})'  # 显示百分比和真实值
    return my_format

# 绘制饼图
# plt.pie(sizes, labels=labels, colors=colors, autopct=autopct_format(sizes), startangle=140)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# 添加图例
plt.legend(labels, title="Categories", loc="upper right")

# 设置饼图为圆形
plt.axis('equal')

# 显示图形
# plt.show()

# save
save_path = '/home/libo/program/l3cprocthor/qxg/datasets/procthor/trajectory/train/pie_chart.jpg'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至：{save_path}")