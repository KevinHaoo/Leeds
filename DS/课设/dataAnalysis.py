import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager, patches
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from sklearn.utils import compute_class_weight

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth',
              1000)
sns.set_theme(style="whitegrid")

# data loading and filtering
data = pd.read_csv("data_640_validated.csv", encoding="ISO-8859-1")
selected_columns = ['A1_1', 'A1_2', 'B2', 'D3', 'F31'] + [f'E{i}' for i in range(1, 29)]
data_filtered = data[selected_columns]
data_filtered = data_filtered.dropna()

# preprocessing
missing_data = data.isnull().sum()
missing_data_summary = pd.DataFrame({
    'Column': missing_data.index,
    'Missing Values': missing_data.values,
    'Percentage Missing': (missing_data.values / len(data)) * 100
}).sort_values(by='Percentage Missing', ascending=False).reset_index(drop=True)
# print(missing_data_summary)

print(data_filtered['B2'].unique())

# encoding
b2_mapping = {
    'No self-isolation/social distancing': 0,
    '1 day': 1,
    '3 days': 1,
    '4 days': 1,
    '5 days': 1,
    '6 days': 1,
    '7 days': 1,
    # '1 day': 'Less than a week',
    # '3 days': 'Less than a week',
    # '4 days': 'Less than a week',
    # '5 days': 'Less than a week',
    # '6 days': 'Less than a week',
    # '7 days': 'Less than a week',
    'More than a week': 2,
    'More than 2 weeks': 3,
    'More than 3 weeks': 4,
    'More than a month': 5
}
data_filtered['B2'] = data_filtered['B2'].map(b2_mapping)
# print(data_filtered['B2'].unique())


# Question.a1
# # Calculate the distribution of B2 and sort by frequency
# b2_distribution = data_filtered['B2'].value_counts().sort_values(ascending=False)
# # plot
# plt.figure(figsize=(12, 8))
# palette = sns.color_palette("ocean", len(b2_distribution))[::-1]
# sns.barplot(x=b2_distribution.index, hue=b2_distribution.index, y=b2_distribution.values,
#             legend=False, palette=palette, width=0.6)
# for index, value in enumerate(b2_distribution.values):
#     plt.text(index, value + 1, str(value), ha='center', fontsize=12)
# plt.title('Distribution of B2 Categories', fontsize=16, fontweight='bold')
# plt.xlabel('B2 Categories', fontsize=12)
# plt.ylabel('Frequency', fontsize=14, rotation=90)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("a1.svg", format="svg")
# plt.show()
#
# # Display the sorted distribution
# b2_distribution_summary = b2_distribution.reset_index()
# b2_distribution_summary.columns = ['B2 Category', 'Frequency']

# print(b2_distribution_summary)

# Question.a2
# selected_columns = ['A1_2', 'B2']
# data_filtered = data[selected_columns]
#
# # 确保数据无缺失
# data_filtered = data_filtered.dropna(subset=['A1_2', 'B2'])
#
# # 对 A1_1 分组统计 B2 的分布
# grouped_data = data_filtered.groupby(['B2', 'A1_2']).size().unstack(fill_value=0)
# # 1. 计算每个 B2 的总频率
# grouped_data['Total'] = grouped_data.sum(axis=1)
# #
# # 2. 按总频率从高到低排序
# grouped_data = grouped_data.sort_values(by='Total', ascending=False).drop(columns=['Total'])
# print(grouped_data)
#
# # 3. 获取排序后的 B2 标签和颜色
# categories = grouped_data.index
# regions = grouped_data.columns
# colors = sns.color_palette("PuBu", len(regions))
#
# # 计算总频率并排序
# grouped_data['Total'] = grouped_data.sum(axis=1)
# sorted_grouped_data = grouped_data.sort_values(by='Total', ascending=False)
#
# # 筛选出总频率小于50的 B2
# low_frequency_data = sorted_grouped_data[sorted_grouped_data['Total'] < 50]
# high_frequency_data = sorted_grouped_data[sorted_grouped_data['Total'] >= 50]
#
# # 移除 Total 列以便绘图
# low_frequency_data = low_frequency_data.drop(columns=['Total'])
# high_frequency_data = high_frequency_data.drop(columns=['Total'])
#
# # 获取颜色
# regions = high_frequency_data.columns
# colors = sns.color_palette("PuBu", len(regions))
#
# # 创建一个图框，包含两部分：主图和子图
# fig, axs = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.5})
#
# # --- 主图 ---
# ax_main = axs[0]
# bottom = np.zeros(len(sorted_grouped_data))
#
# for i, (region, color) in enumerate(zip(regions, colors)):
#     ax_main.bar(
#         sorted_grouped_data.index,
#         sorted_grouped_data[region],
#         bottom=bottom,
#         label=region,
#         color=color,
#         edgecolor='black'
#     )
#     bottom += sorted_grouped_data[region].fillna(0).values
#
# # 绘制红色框标记所有小于 50 的区域
# low_frequency_indices = low_frequency_data.index
# x_min = categories.get_loc(low_frequency_indices[0]) - 0.4  # 最左边
# x_max = categories.get_loc(low_frequency_indices[-1]) + 0.4  # 最右边
# y_max = sorted_grouped_data.loc[low_frequency_indices, 'Total'].max()  # 最大高度
#
# # 添加红色矩形框
# ax_main.add_patch(plt.Rectangle(
#     (x_min, 0),  # 左下角
#     x_max - x_min,  # 宽度
#     y_max,  # 高度
#     fill=False, edgecolor='red', linewidth=2, linestyle='--'
# ))
#
# # 添加箭头指向子图
# ax_main.annotate(
#     'See details below',
#     xy=((x_min + x_max) / 2, y_max),  # 箭头的起点
#     xytext=(5, y_max + 50),  # 箭头的终点（子图标题上方）
#     arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5),
#     fontsize=12, color='red', ha='center'
# )
#
# # 主图美化
# ax_main.set_title('B2 Distribution by Regions (Sorted by Total Frequency)', fontsize=16, fontweight='bold')
# ax_main.set_xlabel('B2 Categories (Sorted)', fontsize=12)
# ax_main.set_ylabel('Frequency', fontsize=12)
# ax_main.set_xticks(range(len(categories)))
# ax_main.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
# ax_main.legend(title='Regions', loc='upper right')
# ax_main.grid(axis='y', linestyle='--', alpha=0.7)
#
# # --- 子图 ---
# ax_sub = axs[1]
# bottom = np.zeros(len(low_frequency_data))
#
# # 绘制低频柱状图
# for i, (region, color) in enumerate(zip(regions, colors)):
#     ax_sub.bar(
#         low_frequency_data.index,
#         low_frequency_data[region],
#         bottom=bottom,
#         label=region,
#         color=color,
#         edgecolor='black'
#     )
#     bottom += low_frequency_data[region].fillna(0).values
#
# # 子图美化
# ax_sub.set_title('Detail View: Total Frequency < 50', fontsize=16, fontweight='bold')
# ax_sub.set_xlabel('B2 Categories', fontsize=12)
# ax_sub.set_ylabel('Frequency', fontsize=12)
# ax_sub.set_xticks(range(len(low_frequency_data.index)))
# ax_sub.set_xticklabels(low_frequency_data.index, rotation=45, ha='right', fontsize=10)
# ax_sub.legend(title='Regions', loc='upper right')
# ax_sub.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig("a2.svg", format="svg")
# plt.show()


# pie chart

# 对 A1_2 分组，统计 B2 的分布
grouped_data = data_filtered.groupby(['A1_2', 'B2']).size().unstack(fill_value=0)

# Use seaborn's Blues color palette
colors = sns.color_palette("Blues", len(grouped_data.columns))

# Create subplots for each A1_2 group
num_pies = len(grouped_data)
cols = 2  # Number of columns in the layout
rows = (num_pies + cols - 1) // cols  # Calculate required rows

# Define autopct function to always display percentages
def autopct_format(pct, all_vals):
    absolute = int(round(pct / 100. * sum(all_vals)))
    return f"{pct:.1f}%"
# Adjust font size for percentages and move the title upward
fig, axs = plt.subplots(rows, cols, figsize=(cols * 9, rows * 9))  # Keep the larger figure size

# Flatten axs for easier indexing
axs = axs.flatten()

# Replot the pie charts with adjustments
for i, (label, row) in enumerate(grouped_data.iterrows()):
    wedges, texts, autotexts = axs[i].pie(
        row,
        autopct=lambda pct: autopct_format(pct, row),
        startangle=90,
        colors=colors,
        radius=1.5,  # Keep larger radius
        wedgeprops=dict(edgecolor='black'),
        pctdistance=0.75  # Keep percentage position consistent
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)  # Reduce font size for percentages

    axs[i].set_title(f"{label}", fontsize=16, fontweight='bold', pad=70)  # Move title upward

# Add a single legend for all plots
fig.legend(
    grouped_data.columns,
    loc="center right",
    title="B2 Categories",
    bbox_to_anchor=(1, 0.5),  # Adjust legend position
    fontsize=12
)

# Remove empty subplots if any
for j in range(num_pies, len(axs)):
    axs[j].axis('off')  # Turn off unused subplots

# Adjust layout to prevent overlap and add space for the legend
fig.suptitle("B2 Distribution by Regions", fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Allocate more space for the legend
plt.savefig("a2_pie_chart.svg", format="svg")
plt.show()


# Question.a3
# matplotlib.use("MacOSX")
#
# # 示例交叉表数据
# cross_tab = pd.crosstab(data['B2'], data['D3'])
#
# # 准备数据
# x_labels = cross_tab.columns  # Game Frequency
# y_labels = cross_tab.index    # Self-Isolation Duration
# x, y = np.meshgrid(range(len(x_labels)), range(len(y_labels)))  # 坐标网格
# z = cross_tab.values          # 数据值
#
# # 启用交互模式
# plt.ion()
#
# # 创建 3D 图形
# fig = plt.figure(figsize=(16, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制 3D 图
# surf = ax.plot_surface(x, y, z, cmap='PuBu', edgecolor='k')
#
# # 设置轴标签
# ax.set_xlabel('Game Frequency', fontsize=12, labelpad=20)  # 调整 X 轴标签距离
# ax.set_ylabel('Self-Isolation Duration', fontsize=12, labelpad=20)  # 调整 Y 轴标签距离
# ax.set_zlabel('Frequency', fontsize=12, labelpad=10)  # 调整 Z 轴标签距离
# # 调整坐标轴标签的距离
# ax.xaxis.labelpad = 30  # X 轴标签距离
# ax.yaxis.labelpad = 30  # Y 轴标签距离
#
# # 设置刻度标签
# ax.set_xticks(range(len(x_labels)))
# ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=10)
# ax.set_yticks(range(len(y_labels)))
# ax.set_yticklabels(y_labels, rotation=30, va='center', fontsize=10)
#
# # 添加颜色条
# fig.colorbar(surf, shrink=0.5, aspect=10)
#
# plt.tight_layout()
#
# # 显示图形，窗口支持交互
# plt.show(block=True)

# ==================================

# histogram
# 数据副本，避免修改原始数据
data_copy = data.copy()

# 按照每个 D3 和 B2 的组合计算频率
grouped_data = data_copy.groupby(['D3', 'B2']).size().reset_index(name='Count')

# 计算每个 D3 的总频率（所有 B2 的频率总和）
d3_total = grouped_data.groupby('D3')['Count'].sum().reset_index(name='Total')

# 合并总频率到分组数据
grouped_data = grouped_data.merge(d3_total, on='D3')

# 对 D3 按总频率排序
grouped_data = grouped_data.sort_values(['Total', 'D3'], ascending=[False, True])

# 获取排序后的 D3 标签顺序
sorted_d3 = grouped_data['D3'].drop_duplicates().tolist()

# 对每个 D3 内部的 B2 按频率降序排序
grouped_data = grouped_data.sort_values(['D3', 'Count'], ascending=[True, False])

# 获取每个 D3 内部的 B2 排序顺序
sorted_b2_by_d3 = grouped_data.groupby('D3')['B2'].apply(list)

# 按分组顺序构建全局 B2 排序
b2_order = []
for b2_list in sorted_b2_by_d3:
    b2_order.extend(b2_list)
b2_order = pd.unique(b2_order)  # 去重，保持分组内顺序

# 设置 D3 和 B2 的分类顺序
data_copy['D3'] = pd.Categorical(data_copy['D3'], categories=sorted_d3, ordered=True)
data_copy['B2'] = pd.Categorical(data_copy['B2'], categories=b2_order, ordered=True)

# 绘图
plt.figure(figsize=(14, 8))
palette = sns.color_palette("ocean", len(b2_order))[::-1]
sns.countplot(data=data_copy, x='D3', hue='B2', palette=palette)

# 设置标题和标签
plt.title('Relationship between Self-Isolation and Game Frequency', fontsize=20, fontweight='bold')
plt.xlabel('Game Frequency (D3)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Self-Isolation Duration (B2)', loc='upper right')
plt.xticks(rotation=30)
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("a3_hist.svg", format='svg', dpi=300)
plt.show()


# Sankey plot
#
# # 取值标签
# self_isolation_labels = [
#     "No self-isolation/social distancing", "1 day", "3 days", "4 days", "5 days",
#     "6 days", "7 days", "More than a week", "More than 2 weeks",
#     "More than 3 weeks", "More than a month"
# ]
#
# game_frequency_labels = [
#     "Almost every week", "Almost everyday", "Every month",
#     "Every week", "Everyday", "Less than every month"
# ]
#
# # 节点标签
# labels = self_isolation_labels + game_frequency_labels
#
# # 模拟数据
# np.random.seed(42)
# source_raw = np.random.choice(self_isolation_labels, size=100, p=np.random.dirichlet(np.ones(len(self_isolation_labels))))
# target_raw = np.random.choice(game_frequency_labels, size=100, p=np.random.dirichlet(np.ones(len(game_frequency_labels))))
# value = np.random.randint(5, 20, size=100)
#
# # 构建 DataFrame
# data = pd.DataFrame({'source': source_raw, 'target': target_raw, 'value': value})
#
# # 按目标标签（右侧标签）计算实际频率总和
# target_counts = data.groupby('target')['value'].sum().sort_values(ascending=False)
# sorted_game_frequency_labels = target_counts.index.tolist()
#
# # 更新右侧标签的映射顺序
# target_mapping = {label: i + len(self_isolation_labels) for i, label in enumerate(sorted_game_frequency_labels)}
#
# # 左侧标签保持顺序
# source_mapping = {label: i for i, label in enumerate(self_isolation_labels)}
#
# # 映射索引
# data['source_index'] = data['source'].map(source_mapping)
# data['target_index'] = data['target'].map(target_mapping)
#
# # 合并所有标签
# sorted_labels = self_isolation_labels + sorted_game_frequency_labels
#
# # 按 source_index 和 target_index 分组统计权重
# flows = data.groupby(['source_index', 'target_index'])['value'].sum().reset_index()
#
# # 创建桑基图
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15, thickness=20,
#         line=dict(color="black", width=0.5),
#         label=sorted_labels
#     ),
#     link=dict(
#         source=flows['source_index'],  # 来源节点索引
#         target=flows['target_index'],  # 目标节点索引
#         value=flows['value']           # 流动权重
#     )
# )])
#
# # 设置标题
# fig.update_layout(
#     title_text="Self-Isolation Duration vs Game Playing Frequency (Right Labels Sorted by Frequency)",
#     font_size=12
# )
#
# # 显示图形
# plt.savefig("a3.svg", format='svg')
# fig.show()


# Question.a4

# # 绘制组合直方图
# # 数据预处理：统计每种 B2 和 F31 的组合频率
# grouped_data = data_filtered.groupby(['B2', 'F31']).size().reset_index(name='Count')
#
# # 计算每种 B2 的总频率
# b2_total = grouped_data.groupby('B2')['Count'].sum().reset_index(name='Total')
#
# # 合并总频率到分组数据
# grouped_data = grouped_data.merge(b2_total, on='B2')
#
# # 对 B2 按总频率排序
# grouped_data = grouped_data.sort_values(['Total', 'B2'], ascending=[False, True])
#
# # 获取排序后的 B2 顺序
# sorted_b2 = grouped_data['B2'].drop_duplicates().tolist()
#
# # 设置 B2 和 F31 的分类顺序
# grouped_data['B2'] = pd.Categorical(grouped_data['B2'], categories=sorted_b2, ordered=True)
# feeling_order = [1, 2, 3, 4, 5]
# grouped_data['F31'] = pd.Categorical(grouped_data['F31'], categories=feeling_order, ordered=True)
#
# # 创建数据副本，仅保留最后 6 个 B2 类别
# last_6_b2 = sorted_b2[-6:]
# detailed_data = grouped_data[grouped_data['B2'].isin(last_6_b2)]
#
# # 创建主图和子图框，调整间距
# fig, axes = plt.subplots(
#     2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.6})
# bar_palette = sns.color_palette('PuBu', n_colors=len(feeling_order))
# line_palette = sns.color_palette('Set1', n_colors=len(feeling_order))
# axes[0].grid(axis='y', linestyle='--', alpha=0.7)
# axes[1].grid(axis='y', linestyle='--', alpha=0.7)
#
# # 主图：全范围堆叠条形图
# sns.barplot(data=grouped_data, x='B2', y='Count', hue='F31', palette=bar_palette, ax=axes[0])
# axes[0].set_title('Comparison of Self-Isolation Duration and Feeling of Losing Connection', fontsize=18, fontweight='bold')
# axes[0].set_xlabel('Self-Isolation Duration (B2)', fontsize=12)
# axes[0].set_ylabel('Frequency', fontsize=12)
# axes[0].tick_params(axis='x', rotation=30, bottom=False)  # 隐藏主图的横坐标标签
# axes[0].legend(title='Feeling: Lost Connection', loc='upper right')
#
# # 在主图中标记最后6个类别区域
# last_6_start = len(sorted_b2) - 6
# rect_height = grouped_data['Count'].max() * 0.1  # 调整红框的高度
# rect = patches.Rectangle(
#     (last_6_start - 0.5, 0), 6, rect_height,
#     linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
# )
# axes[0].add_patch(rect)
# axes[0].annotate(
#     'See Details Below', xy=(last_6_start + 3, rect_height * 1.1),
#     xytext=(last_6_start, rect_height * 2),
#     arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=12, color='red'
# )
#
# # 子图：最后 6 个 B2 的细节展示
# detailed_data = grouped_data[grouped_data['B2'].isin(last_6_b2)].copy()  # 添加 .copy() 明确生成副本
#
# detailed_data.loc[:, 'B2'] = pd.Categorical(detailed_data['B2'], categories=last_6_b2, ordered=True)
#
# sns.barplot(data=detailed_data, x='B2', y='Count', hue='F31', palette=bar_palette, ax=axes[1])
#
# # 设置子图标题和标签
# axes[1].set_title('Detail View: Last 6 Self-Isolation Duration Categories', fontsize=16, fontweight='bold')
# axes[1].set_xlabel('Self-Isolation Duration (B2)', fontsize=12)
# axes[1].set_ylabel('Frequency', fontsize=12)
#
# # 设置横坐标标签，仅显示最后6组 B2
# axes[1].set_xticks(range(len(last_6_b2)))
# axes[1].set_xticklabels(last_6_b2, rotation=45)
#
# # 修正显示范围，使直方完整展示
# axes[1].set_xlim(-0.5, len(last_6_b2) - 0.5)
#
# # 美化子图的边框
# for spine in ['top', 'right']:
#     axes[1].spines[spine].set_visible(True)
#
# # 修正图例位置
# axes[1].legend(title='Feeling: Lost Connection', loc='upper right')
#
# # 调整布局并保存
# fig.subplots_adjust(hspace=0.8)  # 增大主图和子图之间的垂直间距
# # plt.tight_layout()
# plt.savefig("a4_hist.svg", format='svg', dpi=300)
# plt.show()



# # 绘制经典箱线图
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data, x='B2', hue='B2', y='F31', palette='Blues')
#
# # 设置标题和标签
# plt.title('Distribution of Feeling of Losing Connection by Self-Isolation Duration', fontsize=16, fontweight='bold')
# plt.xlabel('Self-Isolation Duration (B2)', fontsize=12)
# plt.ylabel('Feeling: Lost Connection (F31)', fontsize=12)
# plt.xticks(rotation=30, ha='right')
# plt.savefig("a4_basicBoxplot.svg", format='svg')
# plt.tight_layout()
# plt.show()


# # 小提琴箱线图
# plt.figure(figsize=(14, 8))
#
# # 绘制分组的核密度图
# sns.violinplot(data=data, x='B2', hue='B2', y='F31', density_norm='width', inner=None, linewidth=1, cut=0, palette="Blues")
#
# # 叠加箱线图（去除离群点）
# sns.boxplot(data=data, x='B2', hue='B2', y='F31', width=0.25, showfliers=False, linewidth=1.5, boxprops={'zorder': 3}, palette="gist_grey")
#
# # 设置标题和标签
# plt.title('Self-Isolation Duration vs Feeling of Losing Connection', fontsize=16, fontweight='bold')
# plt.xlabel('Self-Isolation Duration (B2)', fontsize=14)
# plt.ylabel('Feeling of Losing Connection (F31)', fontsize=14)
# plt.xticks(rotation=30, ha='right')
#
# plt.tight_layout()
#
# # 导出为矢量图
# plt.savefig("a4_violin_boxplot.svg", format='svg', dpi=300)
# plt.show()




# Question.b

# RF
# 1. 数据准备
X = data_filtered[[f'E{i}' for i in range(1, 29)]]
y = data_filtered['B2']  # 目标变量

# 将目标变量转化为分类标签
y = y.astype('category').cat.codes

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 模型训练
# model = RandomForestClassifier(n_estimators=100, random_state=9)
# model.fit(X_train, y_train)
#
# # 3. 提取特征重要性
# importances = model.feature_importances_
# feature_names = X.columns
# feature_importance = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance (%)': importances * 100
# }).sort_values(by='Importance (%)', ascending=True).reset_index(drop=True)
#
# # 4. 可视化
# plt.figure(figsize=(10, 6))
# # plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
# sns.barplot(x="Importance (%)", y="Feature", hue="Feature", data=feature_importance, palette="Blues")
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.title('Feature Importance (Random Forest)')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()



# XGBoost
# 分离特征和目标变量
X = data_filtered[[f'E{i}' for i in range(1, 29)]]
y = data_filtered['B2']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# 创建和训练XGBoost模型
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
xgb_model.fit(X_train, y_train)

# 计算特征重要性
xgb_feature_importances = xgb_model.feature_importances_
feature_names = X.columns

# 将特征重要性与特征名关联
xgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance (%)': xgb_feature_importances * 100
}).sort_values(by='Importance (%)', ascending=True).reset_index(drop=True)

# 打印重要性排序
print(xgb_importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance (%)", y="Feature", hue="Feature", data=xgb_importance, palette="Blues")
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance calculated by XGBoost')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# SHAP 分析
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
shap_importances = np.abs(shap_values).mean(axis=0)

# 全局特征重要性可视化
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.title("Feature Importance by SHAP")
plt.tight_layout()
plt.show()

# 局部特征重要性可视化
shap.summary_plot(shap_values, X_train, show=False)
plt.title("Local Feature Importance Analysis Using SHAP", fontsize=18, fontweight='bold')  # 添加标题
plt.tight_layout()
plt.savefig('b_shap_summary_with_title.svg', format='svg', dpi=300)
plt.show()

shap_mean_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'SHAP Importance (mean abs, %)': abs(shap_values).mean(axis=0) * 100
}).sort_values(by='SHAP Importance (mean abs, %)', ascending=False)

# 标准化 SHAP 特征重要性
shap_mean_importance['Normalized SHAP Importance (%)'] = (
    shap_mean_importance['SHAP Importance (mean abs, %)'] /
    shap_mean_importance['SHAP Importance (mean abs, %)'].sum()
) * 100

# 合并 XGBoost 和 SHAP 的特征重要性
merged_importance = pd.merge(
    xgb_importance,
    shap_mean_importance[['Feature', 'Normalized SHAP Importance (%)']],
    on='Feature',
    how='outer'
).sort_values(by='Normalized SHAP Importance (%)', ascending=False)

# 打印表格到控制台
print("\nFeature Importance Comparison (XGBoost vs SHAP):")
print(merged_importance)


# 对比可视化
# 创建图表
plt.figure(figsize=(14, 8))

# 绘制 XGBoost 特征重要性（宽条形）
sns.barplot(
    x=merged_importance['Feature'],
    y=merged_importance['Importance (%)'],
    color='#6BAED6',
    alpha=0.8,
    width=0.7,
    label='XGBoost'
)

# 绘制 SHAP 特征重要性（窄条形）
sns.barplot(
    x=merged_importance['Feature'],
    y=merged_importance['Normalized SHAP Importance (%)'],
    color='#D95319',
    alpha=0.9,
    width=0.2,
    linewidth=0,
    label='SHAP'
)
plt.legend(title='Gradient Legend', loc='upper right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Normalized Importance (%)', fontsize=12)
plt.title('Comparison of Feature Importance (XGBoost vs SHAP)', fontsize=20, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('importance comparison.svg', format='svg', dpi=300)
plt.show()



# Question.c
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# 定义 Focal Loss 函数
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(preds, dtrain):
        # 获取标签
        labels = dtrain.get_label()

        # 计算概率（Softmax）
        preds_exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        preds_prob = preds_exp / np.sum(preds_exp, axis=1, keepdims=True)

        # 计算梯度和 Hessian
        grad = preds_prob - np.eye(len(np.unique(labels)))[labels.astype(int)]
        grad = alpha * (1 - preds_prob) ** gamma * grad

        hess = preds_prob * (1 - preds_prob)
        hess = alpha * (1 - preds_prob) ** gamma * (gamma * grad + hess)

        return grad.flatten(), hess.flatten()

    return loss


# 提取特征和目标
X = data_filtered[[f'E{i}' for i in range(1, 29)]]
y = pd.Categorical(data_filtered['B2']).codes

from sklearn.model_selection import StratifiedKFold, train_test_split

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9, stratify=y
)
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=9)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 基于特征重要性排序
feature_importances = [
    'E25', 'E10', 'E16', 'E12', 'E21', 'E17', 'E7', 'E22', 'E14', 'E27',
    'E28', 'E23', 'E6', 'E20', 'E24', 'E9', 'E18', 'E4', 'E26', 'E11',
    'E15', 'E19', 'E5', 'E8', 'E3', 'E1', 'E13', 'E2'
]

# 定义最佳模型参数和变量
best_accuracy = 0
best_model = None
best_features = None
test = None
pred = None
prob = None

# 使用 xgb.train 一致的模型方式
accuracies = []
features_count = []

for i in range(1, len(feature_importances) + 1):
    selected_features = feature_importances[:i]
    X_selected = X_resampled[selected_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    fold_accuracies = []

    for train_index, val_index in skf.split(X_selected, y_resampled):
        X_fold_train, X_fold_val = X_selected.iloc[train_index], X_selected.iloc[val_index]
        y_fold_train, y_fold_val = y_resampled[train_index], y_resampled[val_index]

        # 使用 xgb.train
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_resampled)),
            'eval_metric': 'mlogloss',
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=50,
            # obj=focal_loss(alpha=0.25, gamma=2.0),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # 预测验证集
        y_pred_val = model.predict(dval)
        y_pred_labels = np.argmax(y_pred_val, axis=1)

        # 保存最佳模型
        acc = accuracy_score(y_fold_val, y_pred_labels)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_features = selected_features
            test = y_fold_val
            pred = y_pred_labels
            prob = y_pred_val

        fold_accuracies.append(accuracy_score(y_fold_val, y_pred_labels))

    avg_accuracy = np.mean(fold_accuracies)
    accuracies.append(avg_accuracy)
    print(avg_accuracy)
    features_count.append(i)



# 绘制特征数量与准确率的关系
plt.figure(figsize=(10, 6))
plt.plot(features_count, accuracies, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Average Accuracy')
plt.xticks(range(min(features_count), max(features_count) + 1, 1))  # x轴间隔为1
plt.title('Feature Count vs. Model Accuracy', fontweight='bold', fontsize=20)
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.savefig('comparison different numbers of Ei.svg', format='svg', dpi=300)
plt.show()

# 最佳特征模型评估
X_test_optimal = X_test[best_features]
dtest = xgb.DMatrix(X_test_optimal)
y_proba = best_model.predict(dtest)
y_pred = np.argmax(y_proba, axis=1)

# 分类报告输出
print("Classification Report:")
print(classification_report(test, pred))
print(f"Accuracy: {best_accuracy:.2f}")

# 绘制混淆矩阵
cm = confusion_matrix(test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(data_filtered['B2']))
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Accuracy: {accuracy_score(test, pred):.2f})", fontweight='bold', fontsize=20)
plt.savefig('confusion matrix1.svg', format='svg', dpi=300)
plt.show()


print(f"Shape of y_pred_val: {pred.shape}")
print(pred[:10])  # 打印预测值，查看是否是类别或概率


# 计算 ROC 曲线数据
fpr, tpr, roc_auc = {}, {}, {}
num_classes = len(np.unique(y_resampled))

# y_pred_val: 概率分布
for i in range(num_classes):
    # 将当前类别视为正样本，其余类别为负样本
    y_true_binary = (test == i).astype(int)  # 真实标签二值化
    y_proba_binary = prob[:, i]             # 当前类别的概率

    # 计算 FPR 和 TPR
    fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_proba_binary)
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制 ROC 曲线
plt.figure(figsize=(10, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i] - 0.12:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (best model)', fontweight='bold', fontsize=20)
plt.legend(loc='lower right')
plt.grid()
plt.savefig('ROC1.svg', format='svg', dpi=300)
plt.show()
#
# # 最优特征数
# optimal_features = feature_importances[:np.argmax(accuracies) + 1]
#
# # 确保最优特征数的一致性
# X_train_optimal = X_resampled[optimal_features]
# X_test_optimal = X_test[optimal_features]
#
# # 训练最终模型
# dtrain = xgb.DMatrix(X_train_optimal, label=y_resampled)
# dtest = xgb.DMatrix(X_test_optimal, label=y_test)
#
# params = {
#     'objective': 'multi:softprob',
#     'num_class': len(np.unique(y_resampled)),
#     'eval_metric': 'mlogloss',
# }
#
# final_model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=100,
#     obj=focal_loss(alpha=0.25, gamma=2.0),
#     evals=[(dtrain, 'train'), (dtest, 'test')],
#     early_stopping_rounds=10,
# )
#
# # 预测测试集
# y_proba_test = final_model.predict(dtest)
# y_pred_test = np.argmax(y_proba_test, axis=1)
#
# # 可视化XGBoost树
# xgb.plot_tree(final_model, num_trees=0)
# plt.rcParams['figure.figsize'] = [10, 10]
# plt.savefig('xgb_tree.svg', format='svg', dpi=300)
# plt.show()
#
# accuracy = accuracy_score(y_test, y_pred_test)
#
# # 混淆矩阵
# cm = confusion_matrix(y_test, y_pred_test)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(data_filtered['B2']))
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
# plt.savefig('confusion matrix1.svg', format='svg', dpi=300)
# plt.show()
#
# # ROC曲线
# # y_proba = final_model.predict_proba(X_test_optimal)
#
# fpr, tpr, roc_auc = {}, {}, {}
#
# for i in range(len(np.unique(y))):
#     fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# plt.figure(figsize=(10, 6))
# for i in range(len(np.unique(y))):
#     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid()
# plt.savefig('ROC1.svg', format='svg', dpi=300)
# plt.show()
#
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred_test))
# print(f'Accuracy: {accuracy:.2f}')

# 模型优化方向
# print("""
# Model Optimization Directions:
# 1. Hyperparameter tuning using GridSearchCV or Bayesian Optimization to refine learning rate, max depth, and tree count.
# 2. Data augmentation or class balancing techniques if class distribution is imbalanced.
# 3. Exploring ensemble methods by combining XGBoost with other algorithms for boosting performance.
# 4. Incorporating domain knowledge to engineer additional meaningful features.
# """)
