# Assuming these are your custom modules for fetching and processing papers
import html2text
from tqdm import tqdm,trange
import time
import os
import json
from bs4 import BeautifulSoup
from collections import deque
import numpy as np
import umap

import pickle
from collections import defaultdict

from llm_model import APIModel
from input_load import DataLoader
from multi_gen import GenModel
from Keys import * # SEMANTIC_SCHOLAR_API_KEY,SELECT_MODEL



def assert_keys(pred,ground):
    if len(pred) != len(ground):
        return False
    for key1 in ground.keys():
        if key1 not in pred.keys():
            return False
    return True



def bfs_taxo_old(gen_model, arxiv_id, title, papers, paper_index):
    # 队列中存储 (节点, 子树, 层数)
    queue = deque()
    result = []

    # 将根节点及其层数加入队列
    # for root in tree:
    #     queue.append((root, tree[root], 0))  # 根节点的层数为 0

    # paper_index = paper_index[:5]
    final_cluster, papers = gen_model.dp_clustering(arxiv_id, title, None, papers, paper_index, level=1) 
    for (topic_tmp, papers_c) in final_cluster:
        queue.append((topic_tmp, papers_c, 1)) 


    while queue:
        # 取出队首元素
        topic, papers_c_l, level = queue.popleft()
        print(topic, papers_c_l, level)
        result.append((topic, papers_c_l, level))  # 记录节点及其层数

        if len(papers_c_l)>8 and level<4:        
            # 将子节点及其层数加入队列
            final_cluster, papers = gen_model.dp_clustering(arxiv_id, title, topic, papers, papers_c_l,level=level+1) 
            for (topic_tmp, papers_c_l_tmp) in final_cluster:
                print(topic_tmp, papers_c_l_tmp)
                queue.append((topic_tmp, papers_c_l_tmp, level + 1)) 

    return result


def rename_topic(gen_model,final_cluster,papers,title,topic_root):
    sub_clusters = {}
    for c_index, (topic_tmp, papers_c) in enumerate(final_cluster):
        sub_clusters[topic_tmp+f"_{c_index}"] = [{
            "Title": papers[index]["title"],
            "Abstract": papers[index][topic_tmp]
        } for index in papers_c]

    if len(sub_clusters)>0:
        for i in range(10):
            cluster_info = gen_model.taxo_heading(title, topic_root, sub_clusters)
            print("cluster_info")
            print(final_cluster,cluster_info)
            if assert_keys(cluster_info,sub_clusters):
                return cluster_info
            else:
                print("cluster_info", cluster_info.keys())
                print("sub_clusters", sub_clusters.keys())

    return None
    

def bfs_taxo(gen_model, arxiv_id, title, papers, paper_index):
    # 队列中存储 (节点, 子树, 层数)
    queue = deque()
    result = []

    # 将根节点及其层数加入队列
    # for root in tree:
    #     queue.append((root, tree[root], 0))  # 根节点的层数为 0

    # paper_index = paper_index[:5]
    final_cluster, papers = gen_model.dp_clustering(arxiv_id, title, None, papers, paper_index, level=1) 
    
    cluster_info = rename_topic(gen_model,final_cluster,papers,title,title)

    if cluster_info is not None:               
        for c_index, (topic_tmp, papers_c) in enumerate(final_cluster):
            queue.append((cluster_info[topic_tmp+f"_{c_index}"], papers_c, 1,topic_tmp)) 
    else:
        for (topic_tmp, papers_c) in final_cluster:
            queue.append((topic_tmp, papers_c, 1,topic_tmp)) 

    while queue:
        # 取出队首元素
        topic, papers_c_l, level, topic_dim = queue.popleft()
        print(topic, papers_c_l, level)
        result.append((topic, papers_c_l, level, topic_dim))  # 记录节点及其层数

        if len(papers_c_l)>8 and level<4:        
            # 将子节点及其层数加入队列
            final_cluster, papers = gen_model.dp_clustering(arxiv_id, title, topic, papers, papers_c_l,level=level+1) 
            
            cluster_info = rename_topic(gen_model,final_cluster,papers,title,topic)

            if cluster_info is not None:               
                for c_index, (topic_tmp, papers_c_l_tmp) in enumerate(final_cluster):
                    queue.append((cluster_info[topic_tmp+f"_{c_index}"], papers_c_l_tmp, level + 1, topic_tmp)) 
            else:
                for (topic_tmp, papers_c_l_tmp) in final_cluster:
                    queue.append((topic_tmp, papers_c_l_tmp, level + 1, topic_tmp)) 

    return result



class Node():
    def __init__(self, topic,level=0):
        self.topic = topic
        self.level = level
        self.name = topic.split("_")[0]
        self.paper_list = []
        self.father = None
        self.childern = []
        pass

    def update_name(self, name):
        self.name = name

    def get_name(self):
        return self.name
    
    def update_father(self, f):
        self.father = f

    def get_father(self):
        return self.father
    
    def update_child(self, c):
        self.childern.append(c)

    def get_childern(self):
        return self.childern
    
    def update_paper(self, p):
        self.paper_list.extend(p)

    def get_paper_list(self):
        return self.paper_list


def find_node(node_dic, paper_ids):
    
    for node in reversed(node_dic.values()):
        flag = True
        paper_l = node.get_paper_list()
        for p in paper_ids:
            if p not in paper_l:
                flag = False
        if flag == True and len(paper_l)>len(paper_ids):
            return node
        
    return None




def node_reconstruct(clusters,title):
    node_dic = {}
    root_node = Node(title,level=0)
    root_node.update_name(title)
    # visited_papers = set()

    for c_index, cluster in enumerate(clusters):
        topic = f"{cluster[0]}_{c_index}"
        paper_ids = cluster[1]
        level = cluster[2]

        node_dic[c_index] = Node(topic,level=level)
        node_dic[c_index].update_paper(paper_ids)

        # topic_l = topic.split("->")
        # print(topic_l)
        # if len(topic_l)>1:
        node_f = find_node(node_dic, paper_ids)
        # print("c_papers", paper_ids)
        # print("f_papers",node_f.get_paper_list())

        # else:
        if node_f is None:
            node_f = root_node

        node_c = node_dic[c_index]

        node_f.update_child(node_c)
        node_c.update_father(node_f)
        # print(node_f.topic,node_c.topic)

    return root_node





def rename_tree(gen_model, root_node, title, papers):
    children = root_node.get_childern()  
    sub_clusters = {}
    for c_index, child in enumerate(children):
        paper_l = child.get_paper_list()
        # name = child.get_name()        
        topic = child.topic  #+f"_{c_index}"  
        # child.topic = topic
        # print(name,topic)

        sub_clusters[topic] = [{
                "Title": papers[index]["title"],
                "Abstract": papers[index][topic.split("_")[0]]
            } for index in paper_l]
    
    # print(sub_clusters)
    if len(sub_clusters)>0:
        for i in range(10):
            cluster_info = gen_model.taxo_heading(title, root_node.get_name(), sub_clusters)
            if assert_keys(cluster_info,sub_clusters):
                break
            else:
                print("cluster_info", cluster_info.keys())
                print("sub_clusters", sub_clusters.keys())
                    
        for child in children:     
            topic = child.topic 
            try:
                child.update_name(cluster_info[topic])
            except Exception as e:
                print(e)
            rename_tree(gen_model, child, title, papers)


def taxo_heading(gen_model, arxiv_id, title, papers, paper_index):

    result = []

    def search_tree(root_node):        
        children = root_node.get_childern()
        for child in children:
            result.append(f"{'#'*child.level}  {child.get_name()}")
            result.append(f' {{"Papers": {child.get_paper_list()}}}')

            # print(child.topic, child.get_name(),child.level)
            # print(child.get_paper_list())
            search_tree(child)
    
    final_res = bfs_taxo(gen_model, arxiv_id, title, papers, paper_index)
    
    taxo_tree_root = node_reconstruct(final_res,title)
    # print(final_res)  

    # rename_tree(gen_model, taxo_tree_root, title, papers)
    search_tree(taxo_tree_root)
    # print("\n".join(result))

    # print("\n".join(result))
    return result,final_res




def total_control(aspect_type, dp_type, aspect_k, final_k, n_neighbors, llm_client,noise_ratio=0.1):
    # arxiv_id = "2409.18786"
    
    save_path_root = f"{SAVE_PATH}/{SELECT_MODEL}/c+n-noisy_{noise_ratio}_{aspect_type}-{dp_type}-{aspect_k}-{final_k}-neib{n_neighbors}"
    # save_path_root = f"../res/aws-claude3.7/test"
    # save_path_root = f"{save_res_root}/dp_cluster"
    noise_path_root = f"../data/noise/ratio_{noise_ratio}"
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    same_file = ['2404.00699', '2404.18231', '2410.03751', '2210.12714', '2409.18786', '2406.17624', '2312.11562', '2203.05227', '2407.18369', '2302.01859', '2405.17935', '2106.15561', '2212.09597', '2406.11289', '2312.07913', '2312.01700', '2203.01054', '2405.18653', '2401.01313', '2408.06361', '2212.13465', '2406.01171', '2312.03863', '2403.18105', '2410.22180']
    # same_file = ['2311.05876', '2404.00699', '2306.02051', '2404.18231', '2309.15857', '2410.03751', '2402.18267', '2312.17617', '2410.18529', '2406.01252', '2406.03712', '2410.15576', '2210.12714', '2409.18786', '2411.05902', '2405.12819', '2403.15412', '2406.05615', '2308.12014', '2406.17624', '2312.11562', '2310.14724', '2406.00936', '2401.14656', '2311.12399', '2404.04925', '2407.15017', '2404.01869', '2401.00812', '2407.18418', '2406.10885', '2409.09030', '2403.12027', '2203.05227', '2409.14195', '2407.18369', '2411.11072', '2410.15595', '2405.13019', '2408.10548', '2302.01859', '2310.07343', '2202.08063', '2310.07521', '2309.15402', '2404.01077', '2405.10630', '2404.14851', '2402.09283', '2403.08319', '2311.05232', '2401.07518', '2405.10936', '2308.07633', '2412.03920', '2309.07864', '2405.17935', '2106.15561', '2411.03350', '2303.07616', '2311.07989', '2404.10981', '2402.11291', '2211.03536', '2212.09597', '2309.15025', '2402.01364', '2410.12896', '2305.02750', '2304.11534', '2406.09559', '2312.07622', '2311.07594', '2403.09606', '2406.11289', '2410.15019', '2103.11072', '2312.07913', '2312.01700', '2203.01054', '2404.01230', '2401.15422', '2310.04959', '2307.12966', '2403.01528', '2305.02579', '2407.01603', '2406.06852', '2404.15676', '2311.09008', '2311.03731', '2405.01769', '2311.07914', '2405.18653', '2407.11484', '2409.03752', '2406.06391', '2410.15326', '2409.06857', '2409.07388', '2406.00515', '2310.16218', '2412.02104', '2003.08271', '2403.01152', '2301.00234', '2107.13586', '2407.06204', '2401.01313', '2311.16789', '2403.04786', '2212.13465', '2406.01171', '2212.10535', '2407.04295', '2108.06688', '2312.03863', '2402.01512', '2406.14644', '2403.18105', '2408.01287', '2106.04554', '2209.00099', '2310.15654', '2410.22180']

    ground_path_root = DATA_PATH
    input_loader = DataLoader(SEMANTIC_SCHOLAR_API_KEY, HTML_PATH, MD_PATH)
    gen_model = GenModel(save_path_root, aspect_type, dp_type, aspect_k, final_k, n_neighbors, llm_client)

    for taxo_file in tqdm(reversed(os.listdir(ground_path_root))):
        arxiv_id = taxo_file[:-5].replace("ovo","/")
        # if arxiv_id != "2308.07633":
        #     continue
        if arxiv_id not in same_file:
            continue

        save_path_md = os.path.join(save_path_root, 'heading', f"{arxiv_id}.md")
        save_path_cluster = os.path.join(save_path_root, 'cluster', f"{arxiv_id}.json")
        os.makedirs(os.path.dirname(save_path_md),exist_ok=True)
        os.makedirs(os.path.dirname(save_path_cluster),exist_ok=True)
        if os.path.exists(save_path_cluster):
            print("Exists:", save_path_cluster)
            continue

        db_path = f"{ground_path_root}/{taxo_file}"

        noise_papers_path = os.path.join(noise_path_root, arxiv_id, "papers.json")
        if noise_ratio != 0:
            if os.path.exists(noise_papers_path):
                survey_info = json.load(open(db_path,'r',encoding='utf-8'))
                survey_title = survey_info['title']
                with open(noise_papers_path,"r") as f:
                    papers = json.load(f)
                paper_index = list(range(len(papers)))
            else:
                print("ERROR NOISE DATASET")
                exit(0)
        
        else:
            survey_title, papers, paper_index = input_loader.load_survey_papers_noise(db_path, ratio=noise_ratio)


        # try:           
        results, final_res = taxo_heading(gen_model, arxiv_id, survey_title, papers, paper_index)

        with open(save_path_md,"w") as f:
            f.write("\n".join(results))
        with open(save_path_cluster,"w") as f:
            json.dump(final_res,f,indent=2)
        # except Exception as e:
        #     print(e)

        save_path2 = os.path.join(save_path_root, arxiv_id, "papers.json")
        os.makedirs(os.path.dirname(save_path2),exist_ok=True)

        with open(save_path2,"w") as f:
            json.dump(papers,f,indent=2) 

        with open(os.path.join(save_path_root,"statis.json"),'a',encoding='utf-8') as f:
            statis = {
                "arxiv_id":arxiv_id, 
                "title": survey_title,              
                "dataset": input_loader.arxiv_ratio_list,
                "input":gen_model.client.input_token_count,
                "output":gen_model.client.output_token_count,
                "embedding_token":gen_model.client.embedding_token_count
            }
            
            f.write(json.dumps(statis)+"\n")

            input_loader.arxiv_ratio_list = list()
            gen_model.client.input_token_count = 0
            gen_model.client.output_token_count = 0
            gen_model.client.embedding_token_count = 0

    
    return 


if __name__=="__main__":


    llm_client = APIModel(OPENAI_KEY,OPENAI_URL,SELECT_MODEL,MODEL_PATH, EMBD_MODEL)
    n_neighbors = 6
    for noise_ratio in [0.05]:
        print(f"--------Noise:{noise_ratio}----------")
        total_control(ASPECT_TYPE, DP_TYPE, ASPECT_K, FINAL_K, n_neighbors, llm_client,noise_ratio=noise_ratio)


# multi_aspect_gen()



