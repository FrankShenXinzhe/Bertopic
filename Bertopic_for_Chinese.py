import umap
import torch
import jieba
import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer


# 原始数据（由AI生成的商品评论数据）
sentences = [
    "这个产品真是太棒了，完全超出了我的期望！",  # 好评
    "服务非常差，等了很久都没人理我。",  # 差评
    "商品质量很好，做工精细，值得推荐！",  # 好评
    "物流太慢了，等了一个星期才收到货。",  # 差评
    "店家态度很好，有问题都及时解决了。",  # 好评
    "产品包装很粗糙，感觉像是被拆过。",  # 差评
    "用这个产品感觉效果很明显，非常满意！",  # 好评
    "价格太贵了，感觉不值这个钱。",  # 差评
    "客服回复很快，解决问题也很高效。",  # 好评
    "商品和描述的不符，有点失望。",  # 差评
    "这个牌子我一直都很信任，这次也没让我失望。",  # 好评
    "物流信息更新不及时，让人很焦急。",  # 差评
    "产品质量很好，用起来很顺手。",  # 好评
    "服务态度很差，问个问题都很不耐烦。",  # 差评
    "性价比很高，下次还会再来买。",  # 好评
    "包装太简陋了，一点都不保护商品。",  # 差评
    "这个产品真的很实用，解决了我的大问题。",  # 好评
    "发货速度太慢了，等了好几天才发货。",  # 差评
    "店家还送了小礼物，很贴心。",  # 好评
    "商品有瑕疵，联系客服也不给处理。",  # 差评
    "用这个品牌很多年了，一直都很满意。",  # 好评
    "物流态度很差，商品都摔坏了。",  # 差评
    "产品效果很好，朋友都说我看起来年轻了。",  # 好评
    "价格波动太大，感觉被坑了。",  # 差评
    "店家服务很周到，还给了使用建议。",  # 好评
    "商品颜色和图片不一样，有色差。",  # 差评
    "这个产品真的很方便，省时省力。",  # 好评
    "退货流程太麻烦了，浪费了很多时间。",  # 差评
    "性价比超高，推荐给大家！",  # 好评
    "商品质量一般，不如预期的好。",  # 差评
    "店家发货很快，物流也很给力。",  # 好评
    "客服态度冷漠，问题没得到解决。",  # 差评
    "这个产品用起来很舒服，没有不适感。",  # 好评
    "包装太浪费了，用了很多不必要的材料。",  # 差评
    "效果很好，会继续使用下去的。",  # 好评
    "商品尺寸不合适，也不能退换。",  # 差评
    "店家很诚信，商品和描述的一样。",  # 好评
    "物流太慢了，而且商品还破损了。",  # 差评
    "这个产品真的很值，性价比很高。",  # 好评
    "服务态度很差，以后再也不会来了。",  # 差评
    "用起来很方便，操作也很简单。",  # 好评
    "商品质量太差，用了几次就坏了。",  # 差评
    "店家还提供了发票，很正规。",  # 好评
    "物流信息不准确，让人很困扰。",  # 差评
    "这个产品真的很适合我，很满意。",  # 好评
    "价格不合理，比其他家贵很多。",  # 差评
    "店家服务很好，还给了优惠。",  # 好评
    "商品有异味，让人很不舒服。",  # 差评
    "这个产品效果很好，值得购买。",  # 好评
    "发货速度太慢，等了很久才收到。",  # 差评
    "店家态度很好，还帮忙解决了使用中的问题。",  # 好评
    "商品和图片不一样，有点欺骗消费者。",  # 差评
    "这个产品真的很实用，家庭必备。",  # 好评
    "客服回复太慢，问题得不到及时解决。",  # 差评
]

# 全局变量
cluster_method = 'KMeans'  # 聚类方法（从HDBSCAN和KMeans中选择）
topic_word_num = 10  # 每个主题下主题词的个数

# 导入模型
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-noinstruct')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-noinstruct')
model.eval()

# 计算句嵌入向量
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

# K-means聚类
if cluster_method == 'KMeans':
    sse = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(sentence_embeddings)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, len(sse)+1)], sse, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # 根据手肘法则人工判断聚类个数（根据手肘法则判断出最佳聚类个数为2）
    kmeans = KMeans(n_clusters=int(input('请输入聚类个数：')), random_state=0)
    kmeans.fit(sentence_embeddings)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    df = pd.DataFrame({'sentence': sentences, 'label': labels})

elif cluster_method == 'HDBSCAN':
    # umap降维
    umap_model = umap.UMAP(n_components=2)
    reduced_data = umap_model.fit_transform(sentence_embeddings)

    # HDBSCAN聚类
    clusterer = hdbscan.HDBSCAN()
    labels = clusterer.fit_predict(reduced_data)
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    print('输出结果1：句子——主题对应关系表')
    print(df)

# c-tf-idf
for label in df['label'].unique():

    # 筛选label属性相同的句子至列表
    sentence_class = df[df['label'] == label]['sentence']

    # 分词
    tokenized_sentences = [' '.join(jieba.lcut(sentence)) for sentence in sentence_class]

    # 计算每个词语在每句话中的tf-idf
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()

    # 计算每个词语总的tf-idf值（用于衡量其重要程度）
    word_importance = np.sum(tfidf_array, axis=0)

    # 将词语和其对应的重要性组合成字典
    word_importance_dict = dict(zip(feature_names, word_importance))

    # 根据重要性对词语进行排序，并提取最重要的topic_word_num个词语
    sorted_words = sorted(word_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_words = sorted_words[:topic_word_num]

    # 打印最重要的10个词语及其重要性
    print(f"第{label}类：", end='')
    for word, importance in top_n_words:
        print(f"{word}_{importance}", end="；")
    print("\n")

# umap可视化
label_colors = {
    '-1': 'black',
    '0': 'red',
    '1': 'blue',
    '2': 'green',
    '3': 'yellow',
    '4': 'purple',
    '5': 'orange',
    '6': 'pink',
    '7': 'cyan',
    '8': 'magenta',
    '9': 'brown',
    '10': 'gray',
    '11': 'olive',
    '12': 'teal',
    '13': 'navy',
    '14': 'maroon'
    # 可以继续添加更多标签和颜色
}
umap_model = umap.UMAP(n_components=2)
reduced_data = umap_model.fit_transform(sentence_embeddings)

plt.figure(figsize=(10, 7))
for label in set(df['label']):
    indices = df['label'] == label
    plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=label, c=[label_colors[str(label)]])

plt.title('UMAP Visualization')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.show()

# 计算聚类中心
cluster_centers = {}
for label in set(df['label']):
    indices = df['label'] == label
    cluster_center = np.mean(reduced_data[indices], axis=0)
    cluster_centers[label] = cluster_center

# 计算每个点到其聚类中心的距离
distances = []
for idx, row in df.iterrows():
    label = row['label']
    point = reduced_data[idx]
    center = cluster_centers[label]
    distance = np.linalg.norm(point - center)
    distances.append({'sentence': row['sentence'], 'label': label, 'distance': distance})

# 创建DataFrame并排序
distances_df = pd.DataFrame(distances)
distances_df_sorted = distances_df.sort_values(by='distance')

# 输出排序后的DataFrame（可用于辅助确定各主题的含义）
print(distances_df_sorted)
distances_df_sorted.to_excel('result.xlsx', index=False)
