import pygraphviz as pgv

# 创建一个空图形
G = pgv.AGraph()

# 添加节点，并设置节点属性
G.add_node('A', color='red', style='filled')
G.add_node('B', color='blue', style='filled')
G.add_node('C', color='green', style='filled')

# 添加边，并设置边的属性
G.add_edge('A', 'B', color='black', style='dashed')
G.add_edge('B', 'C', color='black', style='dotted')
G.add_edge('C', 'A', color='black', style='solid')

# 绘制图形并保存为图片
G.layout(prog='dot')
G.draw('custom_graph.png')