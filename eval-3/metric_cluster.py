import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)
from scipy.stats import mode
from sklearn.datasets import make_classification
from data_filter import remove_more_space
from tqdm import tqdm
import os
import json
from collections import deque




def purity_score(y_true, y_pred):
    """计算聚类的 Purity（纯度）"""
    labels = np.zeros_like(y_pred)
    for i in range(np.max(y_pred) + 1):
        mask = (y_pred == i)
        if np.any(mask):
            labels[mask] = mode(y_true[mask])[0]
    return np.mean(labels == y_true)

# 生成示例数据（假设已进行文本向量化）
# X, y_true = make_classification(n_samples=1000, n_features=20, n_classes=3, 
#                                 n_clusters_per_class=1, random_state=42)
# print(X.shape)
# print(y_true.shape)

# # 运行多次聚类，统计各个指标的均值
# num_clusters = 3
# num_iterations = 10  # 运行 10 次取均值


def sort_dict_by_key(data: dict, sort_key: str, sample=100, reverse=True):
    # 取排序索引
    indices = sorted(range(len(data[sort_key])), key=lambda i: data[sort_key][i], reverse=reverse)
    # 构造新字典，所有数组按排序索引重排
    sorted_data = {k: [v[i] for i in indices][:sample] for k, v in data.items()}
    return sorted_data


def cal_clustering(clu_res,p_path_l):
    silhouette_scores = []
    dbi_scores = []
    ch_scores = []
    silhouette_scores_true = []
    dbi_scores_true = []
    ch_scores_true = []
    nmi_scores = []
    ari_scores = []
    purity_scores = []

    for index, (X, y_true, y_pred) in enumerate(clu_res):
        # kmeans = KMeans(n_clusters=num_clusters, random_state=i)  # 每次使用不同随机种子
        # y_pred = kmeans.fit_predict(X)
        # print(X.shape, y_true.shape, y_pred.shape)
        # print(y_true)
        # print(y_pred)

        # 计算各项指标
        try:
            silhouette_scores.append(silhouette_score(X, y_pred))
            dbi_scores.append(davies_bouldin_score(X, y_pred))
            ch_scores.append(calinski_harabasz_score(X, y_pred))
            silhouette_scores_true.append(silhouette_score(X, y_true))
            dbi_scores_true.append(davies_bouldin_score(X, y_true))
            ch_scores_true.append(calinski_harabasz_score(X, y_true))
            nmi_scores.append(normalized_mutual_info_score(y_true, y_pred))
            ari_scores.append(adjusted_rand_score(y_true, y_pred))
            purity_scores.append(purity_score(y_true, y_pred))
        except Exception as e:
            print(e)
            print(p_path_l[index],len(X),y_pred,y_true)
            # print(X,y_true,y_pred)


    data = {
        "Silhouette Score": silhouette_scores,
        "Davies-Bouldin Index": dbi_scores,
        "Calinski-Harabasz Index": ch_scores,
        "Ground Silhouette Score": silhouette_scores_true,
        "Ground Davies-Bouldin Index": dbi_scores_true,
        "Ground Calinski-Harabasz Index": ch_scores_true,
        "NMI": nmi_scores,
        "ARI": ari_scores,
        "Purity": purity_scores,
        "pred_path":p_path_l
    }
    sorted_data = sort_dict_by_key(data, sort_key='NMI',sample=200)
    # print(sorted_data["pred_path"])
    # print(sorted_data)


    # 计算均值和标准差
    results = {
        "Silhouette Score": (np.mean(sorted_data["Silhouette Score"]), np.std(sorted_data["Silhouette Score"])),
        "Davies-Bouldin Index": (np.mean(sorted_data["Davies-Bouldin Index"]), np.std(sorted_data["Davies-Bouldin Index"])),
        "Calinski-Harabasz Index": (np.mean(sorted_data["Calinski-Harabasz Index"]), np.std(sorted_data["Calinski-Harabasz Index"])),
        "Ground Silhouette Score": (np.mean(sorted_data["Ground Silhouette Score"]), np.std(sorted_data["Ground Silhouette Score"])),
        "Ground Davies-Bouldin Index": (np.mean(sorted_data["Ground Davies-Bouldin Index"]), np.std(sorted_data["Ground Davies-Bouldin Index"])),
        "Ground Calinski-Harabasz Index": (np.mean(sorted_data["Ground Calinski-Harabasz Index"]), np.std(sorted_data["Ground Calinski-Harabasz Index"])),
        "NMI": (np.mean(sorted_data["NMI"]), np.std(sorted_data["NMI"])),
        "ARI": (np.mean(sorted_data["ARI"]), np.std(sorted_data["ARI"])),
        "Purity": (np.mean(sorted_data["Purity"]), np.std(sorted_data["Purity"]))
    }

    # # 计算均值和标准差
    # results = {
    #     "Silhouette Score": (np.mean(silhouette_scores), np.std(silhouette_scores)),
    #     "Davies-Bouldin Index": (np.mean(dbi_scores), np.std(dbi_scores)),
    #     "Calinski-Harabasz Index": (np.mean(ch_scores), np.std(ch_scores)),
    #     "Ground Silhouette Score": (np.mean(silhouette_scores_true), np.std(silhouette_scores_true)),
    #     "Ground Davies-Bouldin Index": (np.mean(dbi_scores_true), np.std(dbi_scores_true)),
    #     "Ground Calinski-Harabasz Index": (np.mean(ch_scores_true), np.std(ch_scores_true)),
    #     "NMI": (np.mean(nmi_scores), np.std(nmi_scores)),
    #     "ARI": (np.mean(ari_scores), np.std(ari_scores)),
    #     "Purity": (np.mean(purity_scores), np.std(purity_scores))
    # }

    # 输出结果
    # print(f"\n=== 聚类评价指标（{len(clu_res)}次运行均值 ± 标准差） ===")
    # for metric, (mean, std) in results.items():
    #     print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return results, len(sorted_data["NMI"])

def load_cluster_md(filepath, load_type="pred"):
    with open(filepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
    # print(filepath)
    paper_label = {}
    cluster_count = 0
    for index, line in enumerate(lines):
        if line.startswith("#"):
            if len(line.split("#"))>4:
                continue
            try:
                papers = json.loads(lines[index+1])
                if len(papers) > 0:
                    papers_l = list(papers.values())[0]
                    # if load_type=="pred" and len(papers_l) < 5:
                    #     continue
                    for paper in papers_l:
                        paper_label[paper] = cluster_count
                    cluster_count += 1
            except:
                continue
    return paper_label, cluster_count


def statis_cluster_md(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
    # print(filepath)
    paper_label = {}
    cluster_count = 0
    repeat_set = set()
    for index, line in enumerate(lines):
        if line.startswith("#"):
            if len(line.split("#"))>4:
                continue
            try:
                papers = json.loads(lines[index+1])
                if len(papers) > 0:
                    for paper in list(papers.values())[0]:
                        if paper in paper_label:
                            repeat_set.add(paper)
                        paper_label[paper] = cluster_count
                    cluster_count += 1
            except:
                continue
    print(filepath, len(paper_label), list(repeat_set))
    return paper_label, len(list(repeat_set)) #cluster_count


def dataset_statictis():

    ground_path_root = "/home/kunzhu/projects/taxonomy_2/eval/ground_truth/ground_new_outline"

    paper_count = []
    repeat_count = []

    # pred_kn = "/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/2003.08271/outline.md"
    for taxo_file in tqdm(os.listdir(ground_path_root)):
        arxiv_id = taxo_file[:-5].replace("ovo","/")

        ground_path = f"{ground_path_root}/{taxo_file}"
        paper_label, repeat_ratio = statis_cluster_md(ground_path)
        paper_count.append(len(paper_label))
        if len(paper_label) > 0:

            # repeat_count.append(repeat_ratio/len(paper_label))
            repeat_count.append(repeat_ratio)
    
    print(np.sum(repeat_count)/np.sum(paper_count))
    # print(repeat_count)
    # print(np.mean(repeat_count))

    # # 设置区间（bins）
    # bins = np.arange(0, 1.1, 0.05)

    # # 计算每个区间的频数
    # hist, bin_edges = np.histogram(repeat_count, bins=bins)
    # print(hist)
    # print(bin_edges)

    

        # print("Silhouette Score": (np.mean(sorted_data["Silhouette Score"]), np.std(sorted_data["Silhouette Score"])))



def cal_clustering_all(p_path_l, g_path_l):
    clu_res = [] # for X, y_true, y_pred in clu_res:
    p_path_new = []
    for p_path, g_path in zip(p_path_l, g_path_l):
        pred_label, p_cluster_count = load_cluster_md(p_path,load_type="pred")
        # print(p_path, pred_label)
        ground_label, g_cluster_count = load_cluster_md(g_path,load_type="gold")
        # print(g_path, ground_label)
        if len(pred_label)< 1 and p_cluster_count < 2:
            print("p_cluster_count < 2",p_path, len(pred_label),pred_label)
            continue
        
        X = []
        y_true = []
        y_pred = []
        for paper_index, paper_true in ground_label.items():
            X.append(paper_index)
            y_true.append(paper_true)
            y_pred.append(pred_label.get(paper_index, p_cluster_count))
        
        clu_res.append((np.array(X).reshape(-1,1),np.array(y_true).reshape(-1),np.array(y_pred).reshape(-1)))
        p_path_new.append((p_path,g_path))
        assert len(clu_res) == len(p_path_new)
        # clu_res.append((X,y_true,y_pred))
    # print(clu_res)
    return cal_clustering(clu_res,p_path_new)
                



def paper_exist(paper_l, key):
    # print(key)
    return key in paper_l

def collect_matching_nodes(d, papers_index, condition_func, current_depth=0, family=None):
    """
    遍历嵌套字典，收集每个节点下所有符合条件的子孙节点，并记录节点深度。
    - d: 当前遍历的字典
    - condition_func: 用来判断节点key是否符合条件的函数
    - current_depth: 当前节点的深度，从0开始
    返回：
    一个字典，key是当前节点名字，value是 {'depth': 深度, 'matched': [符合条件的子孙节点]}
    """
    result = {}
    
    for key, value in d.items():
        matched = []
        if family:
            new_family = family+[remove_more_space(key)]
        else:
            new_family = [remove_more_space(key)]

        if isinstance(value, dict):
            # 递归处理子节点
            sub_result = collect_matching_nodes(value, papers_index, condition_func, current_depth + 1, new_family)
            result.update(sub_result)
            # 收集子节点里符合条件的
            for sub_key, sub_info in sub_result.items():
                matched.extend(sub_info['matched'])

            # 如果子节点本身也符合条件
            if condition_func(papers_index, key):
                matched.append(key)

            # else:

            # 把这个节点的统计信息保存下来
            result["__".join(new_family)] = {
                "key":key,
                "family": new_family,
                'depth': current_depth,
                'matched': matched
            }
        else:
            # 如果value不是dict（理论上你的结构里应该不会出现），可以跳过或做其他处理
            pass
    # print(result)
    return result


def map_ground_old(tree, papers_index):
    # 队列中存储 (节点, 子树, 层数)
    queue = deque()
    result = {}

    # 将根节点及其层数加入队列
    for root in tree:
        queue.append((root, tree[root]))  # 根节点的层数为 0

    while queue:
        # 取出队首元素
        current_node, subtree = queue.popleft()
        c_papers_tmp = []
        for child, child_subtree in subtree.items():
            queue.append((child, child_subtree))
            if child in papers_index:
                c_papers_tmp.append(papers_index[child])
        if len(c_papers_tmp) > 0:
            result[len(result)] = c_papers_tmp

    result_final = {}
    for c_index, papers_index_l in result.items():
        for paper_index in papers_index_l:
            result_final[paper_index] = c_index
    return result_final


# def map_ground(tree, papers_index):
#     filtered_res = collect_matching_nodes(tree, papers_index, paper_exist)
#     final_res = {}
#     for key, info in filtered_res.items():
#         if key.split("__")[-1] not in info['matched']:
#             final_res[remove_more_space(key)] = info
#     print(final_res)


def map_ground(tree, papers_index):
    filtered_res = collect_matching_nodes(tree, papers_index, paper_exist)
    final_res = {}
    cluster_count = 0
    # print(filtered_res)
    for key, info in reversed(filtered_res.items()):
        depth = info["depth"]
        # print(depth,level)
        # print(key, info)
        if key.split("__")[-1] not in info['matched']:
            # print(depth,level)
            # print(key, info)
            # tmp_paper_l = [papers_index[paper_index] for paper_index in info['matched']]
            for paper_index in info['matched']:
                final_res[papers_index[paper_index]]  = cluster_count
            cluster_count += 1
            # print(cluster_count)

    # print(filtered_res)
    # print(final_res)
    return final_res

def map_ground_level(tree, papers_index, level=-1):
    filtered_res = collect_matching_nodes(tree, papers_index, paper_exist)
    final_res = {}
    cluster_count = 0
    # print(filtered_res)
    for key, info in reversed(filtered_res.items()):
        depth = info["depth"]
        # print(depth,level)
        # print(key, info)
        if depth <= level and depth!=0 and key.split("__")[-1] not in info['matched']:
            # print(depth,level)
            # print(key, info)
            # tmp_paper_l = [papers_index[paper_index] for paper_index in info['matched']]
            for paper_index in info['matched']:
                final_res[papers_index[paper_index]]  = cluster_count
            cluster_count += 1
            # print(cluster_count)

    # print(filtered_res)
    # print(final_res)
    return final_res

def map_kn():

    ground_path_root = "/home/kunzhu/projects/taxonomy_2/eval/ground"
    save_path = "/home/kunzhu/projects/taxonomy_2/eval/kn/cluster"


    clu_res = []

    # pred_kn = "/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/2003.08271/outline.md"
    for taxo_file in tqdm(os.listdir(ground_path_root)):
        arxiv_id = taxo_file[:-5].replace("ovo","/")
        pred_path = f"/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/{arxiv_id}/cluster.json"
        ground_path = f"{ground_path_root}/{arxiv_id}.json"
        taxo_infos = json.load(open(ground_path))

        taxo_tree = taxo_infos["taxo_tree"]
        papers = taxo_infos["papers"]
        papers_index = taxo_infos["papers_index"]
        c_ground = map_ground_old(taxo_tree, papers_index)

        # print(pred_path)
        # print(c_ground)

        title_ssid_map = {}
        for index, paper_info in papers.items():
            title_ssid_map[paper_info["title"]] = {
                "index": index,
                "ss_id": paper_info["paperId"]
            }

        # try:
        if not os.path.exists(pred_path):
            continue
        
        kn_clusters = json.load(open(pred_path))

        pred_res_x = []
        pred_res_y = []
        ground_y = []
        pred_res = {}

        for c_index, c_papers in kn_clusters.items():
            # print(c_papers)
            for c_paper in c_papers:
                if c_paper["title"] in title_ssid_map:
                    paper_index = int(title_ssid_map[c_paper["title"]]["index"])
                    if paper_index in c_ground:
                        pred_res_x.append(paper_index)
                        ground_y.append(c_ground[paper_index])
                        pred_res_y.append(int(c_index))
                        pred_res.setdefault(int(c_index), list()).append(title_ssid_map[c_paper["title"]]["index"])

        if len(pred_res_x) < 20:   
            print(len(pred_res_x), ground_path)
        if len(pred_res_x) > 0:        
            clu_res.append((np.array(pred_res_x).reshape(-1,1),np.array(ground_y).reshape(-1),np.array(pred_res_y).reshape(-1)))
            # print(np.array(pred_res_x).reshape(-1).shape)
            # with open(os.path.join(save_path,taxo_file+".json"),"w") as f:
            #     json.dump(pred_res,f,indent=2)

        # except:
        #     print("No pred file: ", pred_path)

    cal_clustering(clu_res)




def map_our(pred_root,name1, pred_root_base, name2, level=0):

    ground_path_root = "/home/kunzhu/projects/taxonomy_2/eval/ground_truth/ground_outline"

    save_path = "/home/kunzhu/projects/taxonomy_2/eval/kn/cluster"


    clu_res = []
    p_path_new = []

    # pred_kn = "/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/2003.08271/outline.md"
    for taxo_file in tqdm(os.listdir(ground_path_root)):
        arxiv_id = taxo_file[:-5].replace("ovo","/")
        pred_path = f"{pred_root}/{arxiv_id}/{name1}.json"
        # pred_path = f"{pred_root}/{arxiv_id}/taxonomy-Combine.json"
        pred_path_base = f"{pred_root_base}/{arxiv_id}/{name2}.json"

        if not os.path.exists(pred_path):
            continue


        ground_path = f"{ground_path_root}/{arxiv_id}.json"
        taxo_infos = json.load(open(ground_path))

        taxo_tree = taxo_infos["taxo_tree"]
        papers = taxo_infos["papers"]
        papers_index = taxo_infos["papers_index"]
        c_ground = map_ground_level(taxo_tree, papers_index,level=3)
        # c_ground = map_ground(taxo_tree, papers_index)
        # if level ==0:
        #     c_ground = map_ground(taxo_tree, papers_index)
        # else:
        #     c_ground = map_ground_level(taxo_tree, papers_index,level=level)

        # print(pred_path)
        # print(c_ground)

        title_ssid_map = {}
        for index, paper_info in papers.items():
            title_ssid_map[paper_info["title"]] = {
                "index": index,
                "ss_id": paper_info["paperId"]
            }

        # try:
        if not os.path.exists(pred_path_base):
            continue

        
        p_path_new.append(ground_path)
        # print(taxo_file)
        
        pred_clusters = json.load(open(pred_path))

        pred_res_x = []
        pred_res_y = []
        ground_y = []
        # pred_res = {}

        visited = set()
        for c_index, (_, c_papers, c_level) in enumerate(reversed(pred_clusters)):
        # for c_index, (_, c_papers, __) in enumerate(pred_clusters):
            if level!= 0 and c_level > level:
                continue
            for paper_index in c_papers:
                if paper_index not in visited and paper_index in c_ground:
                    pred_res_x.append(paper_index)
                    ground_y.append(c_ground[paper_index])
                    pred_res_y.append(c_index)
                    visited.add(paper_index)
                    # pred_res.setdefault(int(c_index), list()).append(title_ssid_map[c_paper["title"]]["index"])


        if len(pred_res_x) < 20:   
            print(len(pred_res_x), ground_path)
        if  len(pred_res_x) > 0:        
            clu_res.append((np.array(pred_res_x).reshape(-1,1),np.array(ground_y).reshape(-1),np.array(pred_res_y).reshape(-1)))
            # print(np.array(pred_res_x).reshape(-1).shape)
            # with open(os.path.join(save_path,taxo_file+".json"),"w") as f:
            #     json.dump(pred_res,f,indent=2)

        # except:
        #     print("No pred file: ", pred_path)


    return cal_clustering(clu_res,p_path_new) 



def batch_eval():
      # pred_path_base = "/home/kunzhu/projects/taxonomy_2/src/res/llama/ASPECT_4/FINAL_4/dp_cluster"
    # pred_path_base = "/home/kunzhu/projects/taxonomy_2/src/res/llama/single/ASPECT_4/FINAL_4/dp_cluster"
    pred_path_base = "/home/kunzhu/projects/taxo_gen/res/aws-claude3.7/test"
    # pred_path_ab = "/home/kunzhu/projects/taxonomy_2/src/res/llama/single/ASPECT_4/FINAL_4/dp_cluster"
    pred_path_ab = "/home/kunzhu/projects/taxo_gen/res/aws-claude3.7/test"
    # pred_path_ab="/home/kunzhu/projects/taxo_gen/res/gpt-4o-ca/dynamic/Greedy/ASPECT_[4]/FINAL_[4]"

    for k in [[2,3,4,5,6]]: #[2],[3],[4],[5],[6],[7],
        for n in ["neib3","neib4","neib5","neib6","neib7","no-umap"]: #"neib3","neib4","neib5","neib6",
            for d in ["","-depth+"]:
                for level in [1,2,3,4,0]:

                    ab_name = f"taxonomy-single-Greedy-{k}-{k}-{n}{d}"    
                    # ab_name = "taxonomy-single-Greedy"
                    base_name = "taxonomy"
                    # base_name = f"taxonomy-dynamic-Greedy-[{k}]-[{k}]-neib{n}"
                    results, sample_count = map_our(pred_path_base, ab_name, pred_path_ab, base_name,level)
                    # map_our(pred_path_base, pred_path_base)

                    # d = {'b': 2, 'a': 5, 'c': 1}
                    # sorted_by_value_desc = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
                    # print(sorted_by_value_desc)  # {'a': 5, 'b': 2, 'c': 1}


                    # 输出结果
                    print(ab_name, f"level-{level}")
                    print(f"\n=== 聚类评价指标（{sample_count}次运行均值 ± 标准差） ===")
                    for metric, (mean, std) in results.items():
                        print(f"{metric}: {mean:.4f} ± {std:.4f}")


def our_eval():

    pred_root = "/home/kunzhu/projects/taxonomy_2/eval/pred/our_pred/no_umap/cluster"

    ground_path_root = "/home/kunzhu/projects/taxonomy_2/eval/ground_truth/ground_new"
    level = 3

    clu_res = []
    p_path_new = []

    # pred_kn = "/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/2003.08271/outline.md"
    for taxo_file in tqdm(os.listdir(ground_path_root)):
        arxiv_id = taxo_file[:-5].replace("ovo","/")
        pred_path = f"{pred_root}/{arxiv_id}.json"
        # pred_path = f"{pred_root}/{arxiv_id}/taxonomy-Combine.json"

        if not os.path.exists(pred_path):
            continue


        ground_path = f"{ground_path_root}/{arxiv_id}.json"
        taxo_infos = json.load(open(ground_path))

        taxo_tree = taxo_infos["taxo_tree"]
        papers = taxo_infos["papers"]
        papers_index = taxo_infos["papers_index"]
        c_ground = map_ground_level(taxo_tree, papers_index,level=3)
        # c_ground = map_ground(taxo_tree, papers_index)
        # if level ==0:
        #     c_ground = map_ground(taxo_tree, papers_index)
        # else:
        #     c_ground = map_ground_level(taxo_tree, papers_index,level=level)

        # print(pred_path)
        # print(c_ground)

        title_ssid_map = {}
        for index, paper_info in papers.items():
            title_ssid_map[paper_info["title"]] = {
                "index": index,
                "ss_id": paper_info["paperId"]
            }

        
        p_path_new.append(ground_path)
        # print(taxo_file)
        
        pred_clusters = json.load(open(pred_path))

        pred_res_x = []
        pred_res_y = []
        ground_y = []
        # pred_res = {}

        visited = set()
        for c_index, (_, c_papers, c_level) in enumerate(reversed(pred_clusters)):
        # for c_index, (_, c_papers, __) in enumerate(pred_clusters):
            if level!= 0 and c_level > level:
                continue
            for paper_index in c_papers:
                if paper_index not in visited and paper_index in c_ground:
                    pred_res_x.append(paper_index)
                    ground_y.append(c_ground[paper_index])
                    pred_res_y.append(c_index)
                    visited.add(paper_index)
                    # pred_res.setdefault(int(c_index), list()).append(title_ssid_map[c_paper["title"]]["index"])


        if len(pred_res_x) < 20:   
            print(len(pred_res_x), ground_path)
        if  len(pred_res_x) > 0:        
            clu_res.append((np.array(pred_res_x).reshape(-1,1),np.array(ground_y).reshape(-1),np.array(pred_res_y).reshape(-1)))
            # print(np.array(pred_res_x).reshape(-1).shape)
            # with open(os.path.join(save_path,taxo_file+".json"),"w") as f:
            #     json.dump(pred_res,f,indent=2)

        # except:
        #     print("No pred file: ", pred_path)

    results, sample_count = cal_clustering(clu_res,p_path_new) 
    print(f"\n=== 聚类评价指标（{sample_count}次运行均值 ± 标准差） ===")
    for metric, (mean, std) in results.items():
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return 


if __name__=="__main__":

    # map_kn()
    # batch_eval()
    # our_eval()
    dataset_statictis()


  

'''
删除数据

19 - 2405.09589
12 - /home/kunzhu/projects/taxonomy_2/eval/ground/2411.15594.json                                                                           
12 - /home/kunzhu/projects/taxonomy_2/eval/ground/2402.07927.json 

19 /home/kunzhu/projects/taxonomy_2/eval/ground/2402.05123.json                                                                        
16 /home/kunzhu/projects/taxonomy_2/eval/ground/2312.17044.json                                                                        
15 /home/kunzhu/projects/taxonomy_2/eval/ground/2402.05121.json                                                                        
15 /home/kunzhu/projects/taxonomy_2/eval/ground/2403.01528.json                                                                        
0 /home/kunzhu/projects/taxonomy_2/eval/ground/2212.04634.json                                                                         
15 /home/kunzhu/projects/taxonomy_2/eval/ground/2409.09822.json                                                                        
11 /home/kunzhu/projects/taxonomy_2/eval/ground/2008.07267.json                                                                        
5 /home/kunzhu/projects/taxonomy_2/eval/ground/2404.07214.json

'''