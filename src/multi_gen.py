# 📄 **论文标题**: [Paper Title]  
# ✍ **作者**: [Authors]  
# 📰 **发表会议/期刊**: [Venue]  
# 📅 **发表时间**: [Year]  
# 🎯 **研究问题**: [What problem does it address?]  
# 🔑 **主要贡献**: [Key Contributions]  
# 📊 **方法**: [Methodology Summary]  
# 📈 **实验**: [Main Experimental Results]  
# ⚠ **局限性**: [Limitations]  
# 💡 **个人点评**: [Strengths, Weaknesses, Future Directions]  
# 🔗 **代码/资源**: [GitHub Link / Dataset]  

# from Keys import *
# from openai import OpenAI
# client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_URL)
import hashlib
import time
from itertools import combinations
from tqdm import trange
import json
import os
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering,KMeans,OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import umap
from tqdm import tqdm
from scipy.special import softmax 

from Prompt import *


class GenModel:

    
    def __init__(self, save_res_root, aspect_type, dp_type, aspect_k, final_k, n_neighbors, llm_client) -> None:
        # self.db_path = db_path
        self.save_res_root = save_res_root
        self.client = llm_client
        self.dp_type = dp_type
        self.aspect_type = aspect_type
        self.final_k = final_k
        self.aspect_k = aspect_k
        self.n_neighbors = n_neighbors

    def __assert_aspects(self, aspect_key, gen_keys):
        aspect_key_match = 0
        for key in aspect_key:
            if key in gen_keys:
                aspect_key_match += 1
        return aspect_key_match == len(aspect_key)


    def __check_if_embedding_exists(self, output_dir):
        # Check if the embeddings exist in the output directory
        if os.path.exists(f"{output_dir}_embeddings.pkl"):
            return True
        else:
            return False

    def __embed_and_save_papers_with_openai(self, paper,aspect_key, output_dir):
        data = []
        # print(papers)
        data_for_embedding = []

        if self.dp_type == "Combine":
            combine_summ_l = [f"Title: {paper['title'].strip()}"]
            for key in aspect_key:
                combine_summ_l.append(f'{key}: {paper[key]}')
            combine_summ = "\n".join(combine_summ_l)
            data_for_embedding.append("\n".join(combine_summ))
            response = self.client.embedding_api(data_for_embedding)
            text_embedding_tuplist = [(paper['title'], "Combine", combine_summ, np.array(embedding_obj.embedding)) for embedding_obj in response]
        
        else:
            aspect_key_new = []
            for key in aspect_key:
                if key in paper:
                    if self.aspect_type == "dynamic_new":
                        data_for_embedding.append(f"Title: {paper['title'].strip()} ;\nAbstract: {paper['abstract'].strip()} ;\n{key}: {paper[key]}")
                    else:
                        data_for_embedding.append(f"Title: {paper['title'].strip()} ; {key}: {paper[key]}")
                    aspect_key_new.append(key)
            # print(data_for_embedding)
            response = self.client.embedding_api(data_for_embedding)
            text_embedding_tuplist = [(paper['title'], key, paper[key], np.array(embedding_obj.embedding)) for key, embedding_obj in zip(aspect_key_new, response)]

            # print(222)
            # text_embedding_tuplist = [(text['title'], text['abstract'],text['link'], np.array(embedding_obj.embedding)) for text, embedding_obj in zip(data, response.data)]


        # if not os.path.exists(f'{output_dir}'):
        #    os.makedirs(f'{output_dir}')
        pickle_output_path = f'{output_dir}_embeddings.pkl'
        # Save the data to a file
        with open(pickle_output_path, 'wb') as f:
            pickle.dump(text_embedding_tuplist, f)

        # st.write(f"Text Embedding Done!")
        return text_embedding_tuplist


    def __load_embeddings(self, output_dir):
        # Load the embeddings from the file
        with open(os.path.join(f"{output_dir}_embeddings.pkl"), 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings


    def __extract_data_for_clustering(self, data,top_k=1000):

        # print(len(data),np.min([top_k,len(data)]))
        data = data[0:np.min([top_k,len(data)])]
        # print(data)
        # Extract vectors, titles, and abstracts
        vectors = [item[-1] for item in data]
        # titles_abstracts = [(item[0], item[1]) for item in data]
        titles_abstracts = [{'title': item[0], item[1]: item[2]} for item in data]

        # Convert list of vectors to a numpy matrix
        vector_matrix = np.array(vectors)
        # print(vector_matrix)

        return vector_matrix, titles_abstracts


    def __cluster_papers(self, embeddings, titles_abstracts, n_clusters, cluster_method, is_umap, do_bic,do_silhouette):


        if is_umap:
            # print((len(embeddings) - 1), (len(embeddings) - 1) ** 0.5)
            # n_neib = np.min([int((len(embeddings) - 1) ** 0.5) ,3])

            embeddings = umap.UMAP(n_neighbors=self.n_neighbors  , #50
                                n_components=min(20,len(embeddings)-2), #20
                                min_dist=0,
                                metric='cosine',
                                random_state=42
                                ).fit_transform(embeddings)

        if cluster_method == 'Kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, n_init=10)

        elif cluster_method == 'HCL':  # Renamed from 'HCl' for consistency
            clustering_model = AgglomerativeClustering(
                                                    # metric='cosine',
                                                    distance_threshold=2,
                                                    n_clusters=None)
        elif cluster_method == 'OPTICS':
            clustering_model = OPTICS(metric='cosine', min_samples=2)

        elif cluster_method == 'GMM':
            covariance_type = 'tied' #{'full', 'diag', 'tied', 'spherical'}
            threshold = 0.1
            min_clusters = 2
            reg_covar = 1e-2
            # Select the optimal number of clusters based on BIC
            bic_scores = []
            silhouette_scores = []
            optimal_n_clusters = n_clusters
            # cluster_range = range(min_clusters, n_clusters + min_clusters)
            cluster_range = range(2, 4)
            #cluster_range = range(1, n_clusters*2 + 1)
            print(f"cluster_range: {min_clusters} - {n_clusters + min_clusters}")
            for n in tqdm(cluster_range):
                gmm = GaussianMixture(n_components=n,
                                    random_state=42,
                                    reg_covar=reg_covar,
                                    covariance_type=covariance_type)
                gmm.fit(embeddings)
                if do_bic:
                    bic_scores.append(gmm.bic(embeddings))
                if do_silhouette and n > 1:  # Silhouette score requires at least 2 clusters to be meaningful
                    cluster_labels = gmm.predict(embeddings)
                    silhouette_scores.append(silhouette_score(embeddings, cluster_labels))

            # # Select the optimal number of clusters based on BIC or silhouette
            if do_bic:
                optimal_n_clusters = cluster_range[np.argmin(bic_scores)]
                print(f"Optimal number of clusters based on BIC (GMM): {optimal_n_clusters}")
            if do_silhouette:
                optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
                print(f"Optimal number of clusters based on silhouette score (GMM): {optimal_n_clusters}")


            # Create the final model with the selected number of clusters
            clustering_model = GaussianMixture(n_components=optimal_n_clusters,
                                            covariance_type=covariance_type,
                                            reg_covar=reg_covar,
                                            random_state=42)

            print(f"Optimal number of clusters (GMM): {optimal_n_clusters}")

        else:  # Handle invalid clustering methods
            raise ValueError("Invalid clustering method specified")

        clustering_model.fit(embeddings)
        if cluster_method =='GMM':
            probabilities = clustering_model.predict_proba(embeddings)
            return probabilities, clustering_model.bic(embeddings), clustering_model.predict(embeddings)
        #     cluster_assignment =  [np.where(p > threshold)[0] for p in probabilities]

        else:
            cluster_assignment = clustering_model.labels_
            cluster_dis = clustering_model.inertia_
            centers = clustering_model.cluster_centers_

            # 计算每个样本到每个中心的欧氏距离
            dists = np.linalg.norm(embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

            # 将距离转为“相似度”，然后 softmax（负距离更大相似度）
            similarities = -dists  # 取负号，离得越近值越大
            # print(similarities)
            softmax_probs = softmax(similarities, axis=1)
            return softmax_probs, cluster_dis, cluster_assignment

        #   clusters = dict()
        #   for paper_id,cluster_names in enumerate(cluster_assignment):
        #     for cluster in cluster_names:
        #         if cluster not in clusters:
        #           clusters[cluster] = []

        #         clusters[cluster].append(titles_abstracts[paper_id])

        #   clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]),reverse=True))

        #   # Preparing output
        #   cluster_output = {}
        #   for cluster_id, papers in clusters.items():
        #     # cluster_output[str(cluster_id)] = [{'title': paper[0], 'abstract': paper[1],'link':paper[2]} for paper in papers]
        #     # cluster_output[str(cluster_id)] = [{'title': paper[0], 'abstract': paper[1]} for paper in papers]
        #     cluster_output[str(cluster_id)] = [paper for paper in papers]



        #   return cluster_output,embeddings,cluster_assignment


    def __run_cluster_subtopics(self, emeddings, n_clusters):
        """
        Runs the clustering of subtopics.
        """
        # Load the data
        # if not os.path.exists(f"{output_dir}/cluster.json"):
        
        vector_matrix, titles_abstracts = self.__extract_data_for_clustering(emeddings,top_k=1000)
        # print("vector_matrix:", len(vector_matrix))
        # n_clusters = len(vector_matrix) // 20 # 10 Max number of clusters is 10% of the number of papers
        cluster_method = 'Kmeans'
        is_umap = False
        do_bic = False
        do_silhouette = True
        # # Cluster the papers
        # cluster_output,embeddings_after_cluster,cluster_assignment = cluster_papers(vector_matrix,
        #                                                                             titles_abstracts,
        #                                                                             n_clusters,
        #                                                                             cluster_method,
        #                                                                             is_umap,
        #                                                                             do_bic,
        #                                                                             do_silhouette)
        probabilities, cluster_dis, cluster_assignment = self.__cluster_papers(vector_matrix,
                                    titles_abstracts,
                                    n_clusters,
                                    cluster_method,
                                    is_umap,
                                    do_bic,
                                    do_silhouette)

        # with open(f"{output_dir}/cluster.json","w") as f:
        #     json.dump(cluster_output, f,indent=2)

        # else:
        #     with open(f"{output_dir}/cluster.json") as f:
        #         cluster_output = json.load(f)

        return probabilities, cluster_dis, cluster_assignment, vector_matrix



    def __generate_prompt(self, paras, level=1):
        if level==1:
            user_prompt = generate_aspect_prompt_high_user
            sys_promtp = generate_aspect_prompt_high_sys
            # for k in paras.keys():
                # print(k.capitalize())
        else:
            user_prompt = generate_aspect_prompt_specific_user
            sys_promtp = generate_aspect_prompt_specific_sys

        for k in paras.keys():
            user_prompt = user_prompt.replace(f'[{k.upper()}]', paras[k])

        # user_prompt = user_prompt.replace(f'[TOPIC]', f'"{topic.split("__")[0].replace("_",": ")}"')
        # user_prompt = user_prompt.replace(f'[TITLE_LIST]', "\n - ".join(title_list))

        return sys_promtp, user_prompt


    # def generate_prompt_aspect(paras):
    #     prompt = extract_aspect_prompt
        
    #     for k in paras.keys():
    #         # print(k.capitalize())
    #         prompt = prompt.replace(f'[{k.upper()}]', f'"{paras[k]}"')
    #     return prompt

    def __generate_apspects(self, paras, outpath_root, level=1, cache=True):
        # print(outpath_root)
        sys_p, user_p = self.__generate_prompt(paras, level=level)
        cache_key = hashlib.sha1(user_p.encode()).hexdigest()

        if not os.path.exists(outpath_root):
            os.makedirs(outpath_root)

        if "topic" in paras or len(paras.get("topic","")) > 0:
            outpath = os.path.join(outpath_root, f'{paras["title"]}_{paras.get("topic","")}_{cache_key}.json')
        else:
            outpath = os.path.join(outpath_root, f'{paras["title"]}.json')

        print(outpath_root,outpath)

        aspects_new = {}
        if os.path.exists(outpath) and cache:
            aspects_new = json.load(open(outpath))
            # print(aspects_new)

        else:            

            aspects = self.client.req(sys_p, user_p)
            while aspects is None:
                aspects = self.client.req(sys_p, user_p)
                print("Re-generate aspects!")


            if "topic" in paras:                
                for key, value in aspects.items():
                    aspects_new[f'{paras["topic"]}->{key}']=value
            else:
                aspects_new = aspects
            
            with open(outpath,'w',encoding='utf-8') as f:
                json.dump(aspects_new,f,indent=2)

            # print(aspects_new)

        return aspects_new


    def __generate_multi_abs(self, aspects, paras,outpath, level=1, cache=True):

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        if os.path.exists(outpath) and cache:
            multi_abs = json.load(open(outpath))

        else:
            prompt = extract_aspect_prompt_user
            prompt = prompt.replace("ASPECTS", str(aspects))
            for k in paras.keys():
                # print(k.capitalize())
                prompt = prompt.replace(f'[{k.upper()}]', f'"{paras[k]}"')
            # print(111)
            # response = client.chat.completions.create(
            #         model="gpt-4o-ca",
            #         response_format={ "type": "json_object" },
            #         messages=[
            #             {"role": "system", "content": extract_aspect_prompt_sys.strip()},
            #             {"role": "user", "content":  f"{prompt}"}
            #         ]
            #         )
            # multi_abs = json.loads(response.choices[0].message.content)
            # print(prompt)
            multi_abs = self.client.req(extract_aspect_prompt_sys.strip(), prompt)

            # print(222)
            with open(outpath,'w',encoding='utf-8') as f:
                json.dump(multi_abs,f,indent=2)

        return multi_abs

    def __generate_embeddings(self, paper, aspect_key, output_dir, cache=True):
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        is_embedding_exists = self.__check_if_embedding_exists(output_dir)
        if not is_embedding_exists or not cache:
            papers_and_embeddings = self.__embed_and_save_papers_with_openai(paper,aspect_key,output_dir)
        else:
            papers_and_embeddings = self.__load_embeddings(output_dir)
            # st.write(f"Embedding {len(papers_and_embeddings)} papers about {arxiv_id} {paper["title"]}")
        return papers_and_embeddings


    def __max_weight_selection(self, W, l):
        n, k = W.shape  # 获取点数 n 和选项数 k
        
        numbers = list(range(k)) #0,k-1
        all_sets = list(combinations(numbers, l))
        #print(all_sets)
        print(len(all_sets))
        
        score = {s: 0 for s in all_sets}
        path = {s: [] for s in all_sets}
        
        for i in range(n):
            for s in score.keys():
                tmp_w = 0
                tmp_ind = None
                for j in s:
                    if tmp_w < W[i][j]:
                        tmp_w = W[i][j]
                        tmp_ind = j
                score[s] = score[s] + tmp_w
                path[s].append(tmp_ind)
        
        max_value = 0
        best_path = None
        for s in score.keys():
            if max_value < score[s]:
                max_value = score[s]
                best_path = path[s]

        best_path_new = [(p_index, c_index)for p_index, c_index in enumerate(best_path)]
        
        return max_value, best_path_new

        # for paper in papers:
        #     print(paper)

    def cluster_control(self, embedding_multi, n_clusters, final_k, dp_type, papers_l):
        # n_clusters = self.aspect_k 
        
        final_cluster = []
        final_cluster_index = {}

        if dp_type == "Greedy":
            best_inertia = -np.inf
            best_aspect_key = ""
            best_assignment = []

            for n_cluster in n_clusters:
                for aspect_key, embeddings in embedding_multi.items():
                    # if aspect_key == "abstract" or aspect_key == "introduction":
                    #     continue
                    # n_clusters = len(embeddings) // 20      
                    probabilities, cluster_dis, cluster_assignment, vector_matrix = self.__run_cluster_subtopics(embeddings,n_cluster)
                    sil_score = silhouette_score(vector_matrix, cluster_assignment)
                    if sil_score > best_inertia:
                        best_inertia = sil_score
                        best_aspect_key = aspect_key
                        best_assignment = cluster_assignment
            
            print("final-sil_score:", best_inertia)
            print("final-分配角度:", best_aspect_key)
            print("final-最优方案:", best_assignment)
            
            for paper_index, cluster_index in enumerate(best_assignment):
                final_cluster_index.setdefault(cluster_index, list()).append(papers_l[paper_index])

            for cluster_index, paper_index_l in final_cluster_index.items():
                # final_cluster[cluster_aspect_l[cluster_index]] = 
                # if "topic" in paras:
                #     topic_new = f'{paras["topic"]}->{best_aspect_key}'
                # else:
                topic_new = f'{best_aspect_key}'
                
                final_cluster.append((topic_new,paper_index_l))
        
        else:
            l_list=final_k
            best_inertia = -np.inf
            best_cluster_aspect_l = {}
            best_path_final = []

            for n_cluster in n_clusters:
                cluster_aspect_l = []
                probabilities_all = []
                for aspect_key, embeddings in embedding_multi.items():
                    # if aspect_key == "abstract" or aspect_key == "introduction":
                    #     continue
                    # n_clusters = len(embeddings) // 20      
                    cluster_aspect_l.extend([aspect_key]*n_cluster)
                    probabilities, cluster_dis, cluster_assignment, vector_matrix = self.__run_cluster_subtopics(embeddings,n_cluster)
                    probabilities_all.append(probabilities)                 
                    # print(probabilities)
                #     print(probabilities.shape)  
                #     print(cluster_dis, cluster_assignment)  
                # print(np.hstack(probabilities_all).shape)  

                for l in l_list:
                    if l>n_cluster:
                        continue
            
                    result, best_path = self.__max_weight_selection(np.hstack(probabilities_all), l)
                    print(f"Optimal number of clusters: {n_cluster}-{l}")
                    print("最大权重总和:", result)
                    print("最优方案:", best_path)

                    sil_score = silhouette_score(vector_matrix, [c_index for p_index, c_index in best_path])
                    print("silhouette_score:", sil_score)
                    if sil_score > best_inertia:
                        best_inertia = sil_score
                        best_cluster_aspect_l = cluster_aspect_l
                        best_path_final = best_path

            print("final-sil_score:", best_inertia)
            print("final-分配角度:", best_cluster_aspect_l)
            print("final-最优方案:", best_path_final)
            for paper_index, cluster_index in best_path_final:
                final_cluster_index.setdefault(cluster_index, list()).append(papers_l[paper_index])
            
            for cluster_index, paper_index_l in final_cluster_index.items():
                # final_cluster[cluster_aspect_l[cluster_index]] = 
                # if "topic" in paras:
                #     topic_new = f'{paras["topic"]}->{best_cluster_aspect_l[cluster_index]}'
                # else:
                topic_new = f'{best_cluster_aspect_l[cluster_index]}'
                
                final_cluster.append((topic_new,paper_index_l))
        
        return final_cluster

        
    def __multi_aspect_gen(self, papers, papers_l, paras, output_aspect_root, output_multi_abs_root, output_embedding_root, level=1):

        if self.aspect_type == "single":
            aspects = ["abstract"]
        # elif self.dp_type == "Combine":
        #     aspects = ["Combine"]
        elif self.aspect_type == "fixed":
            aspects =  {
                "Research Problem": "A brief statement of the problem addressed in this study and its significance.",
                "Key Contributions": "A summary of the main innovations and improvements introduced by this study.",
                "Method": "A concise summary of the methodological approach employed in the study",
                "Datasets": "The datasets used in the study, their sources, and their characteristics (size, type, domain).",
                "Experimental Setup": "Key details of the experiment, including training strategies, hyperparameter tuning, hardware setup, and baseline implementations.",
                "Evaluation Metrics": "The metrics used to assess performance (e.g., accuracy, BLEU, ROUGE, F1-score, MSE).",
                "Results & Findings": "Summary of the main experimental outcomes and how they compare with state-of-the-art methods."
                }
        else:
            aspects = self.__generate_apspects(paras, output_aspect_root, level=level, cache=True)
            while aspects is None:
                print("Try to re-generate multi aspects!")
                aspects = self.__generate_apspects(paras, output_aspect_root, level=level, cache=False)
        
        
        embedding_multi = {}
        if self.aspect_type == "single":
            for aspect_key in aspects:
                embedding_multi[aspect_key] = []
        elif self.dp_type == "Combine":
            embedding_multi[self.dp_type] = []
        else:
            for aspect_key in aspects.keys():
                embedding_multi[aspect_key] = []

        
        for p_index, paper in enumerate(tqdm(papers)):

            if paper["arxiv_id"] is not None:
                f_id = paper["arxiv_id"].replace("/","ovo")
            else:
                f_id = paper["paperId"]

            multi_summ_cache = True
            if self.aspect_type == "single":
                output_embedding = os.path.join(output_embedding_root+"_single", f"{paras['title']}", f'{f_id}')

            # elif self.aspect_type == "combine":
            #     output_embedding = os.path.join(output_embedding_root+"combine", f"{paras['title']}", f'{f_id}')

            else:
                # topic_tmp = paras.get('topic','').split("_")[0].split("->")[0]
                topic_tmp = paras.get('topic','')
                save_tmp = os.path.join(output_multi_abs_root,f"{paras['title']}_{topic_tmp}", f'{f_id}_multi.json')
                # print(save_tmp,aspects)
                multi_aps = self.__generate_multi_abs(aspects, paper, save_tmp, level=level)
                # print(aspects.keys(), multi_aps.keys())
                while type(multi_aps) != dict or not self.__assert_aspects(aspects.keys(), multi_aps.keys()):
                    print("Try to re-generate multi aspects summary!", save_tmp)
                    print(multi_aps)
                    multi_summ_cache = False
                    multi_aps = self.__generate_multi_abs(aspects, paper, save_tmp, level=level, cache=False)
                # print(multi_aps)
                if self.aspect_type != "dynamic_new":
                    for a_key, a_summary in multi_aps.items():
                        if "Not applicable" in a_summary:
                            multi_aps[a_key] = paper["title"]                                        
                    
                paper.update(multi_aps)
                # print("Paper Update", multi_aps.keys(), paper.keys())
            # print(time.time()-start)
                output_embedding = os.path.join(output_embedding_root, f"{paras['title']}_{topic_tmp}", f'{f_id}')
            # else:
            #     output_embedding = os.path.join(output_embedding_root, f"{paras['title']}", f'{f_id}')

            if self.dp_type == "Combine":
                output_embedding = os.path.join(output_embedding_root+"Combine", f"{paras['title']}", f'{f_id}')

            embedding_l = self.__generate_embeddings(paper, aspects, output_embedding, cache=multi_summ_cache)

            for paper_tuplist in embedding_l: #(paper['title'], key, paper[key], np.array(embedding_obj.embedding))
                embedding_multi[paper_tuplist[1]].append(paper_tuplist)


        final_cluster =  self.cluster_control(embedding_multi, self.aspect_k, self.final_k, self.dp_type, papers_l)

        return final_cluster, papers


    def dp_clustering(self, arxiv_id, title, topic, papers, papers_c_l, level=1):
        output_aspect_root = f"{self.save_res_root}/{arxiv_id.replace('/','ovo')}/survey_aspect"
        output_embedding_root = f"{self.save_res_root}/{arxiv_id.replace('/','ovo')}/embedding_multi_abs"
        output_multi_abs_root = f"{self.save_res_root}/{arxiv_id.replace('/','ovo')}/arxiv_multi_abs"

        title_list = []
        papers_l = []
        papers_c = []
        for index, paper in enumerate(papers):
        #     # print(paper["arxiv_id"],paper["title"])
            if index in papers_c_l:
                title_list.append(paper["title"])
                papers_l.append(index)
                papers_c.append(paper)
        # print(papers_l)
        # print(papers_c_l)

        paras = {
                "title": title,
                "title_list": "\n - ".join(title_list)
            }
        
        # for p in papers_c:
        #     print(p.keys())
        if topic is not None:
            # print("Topic", topic)
            paras["topic"] = topic
            paras["papers"] = "\n - ".join([f'{p["title"]}: {p[topic]}' for p in papers_c])
                    # user_prompt = user_prompt.replace(f'[TOPIC]', f'"{topic.split("__")[0].replace("_",": ")}"')
        # user_prompt = user_prompt.replace(f'[TITLE_LIST]', "\n - ".join(title_list))
        final_cluster, papers_c = self.__multi_aspect_gen(papers_c, papers_c_l, paras, output_aspect_root, output_multi_abs_root, output_embedding_root, level=level)
        
        return final_cluster, papers
    
    def heading(self, title, topic, papers, level):

        prompt = gen_heading_prompt_user

        prompt = prompt.replace("TITLE", title)
        prompt = prompt.replace("TOPIC", topic)
        if level == 1:
            papers_tmp = [{
                "Title": paper["title"],
                "Abstract": paper[topic]
            } for paper in papers]
        else:
            papers_tmp = [{
                "Title": paper["title"],
                "Abstract": paper["abstract"]
            } for paper in papers]


        prompt = prompt.replace("[PAPERS]", json.dumps(papers_tmp,indent=2))

        heading = self.client.req(gen_heading_prompt_sys, prompt)

        # print(222)
        # with open(outpath,'w',encoding='utf-8') as f:
        #     json.dump(multi_abs,f,indent=2)

        return heading
    
    def taxo_heading(self, title, topic, sub_clusters):

        prompt = gen_taxonomy_heading_prompt_user

        prompt = prompt.replace("TITLE", title)
        prompt = prompt.replace("TOPIC", topic)

    
        prompt = prompt.replace("[PAPERS]", json.dumps(sub_clusters,indent=2))

        heading = self.client.req(gen_taxonomy_heading_prompt_sys, prompt)

        # print(222)
        # with open(outpath,'w',encoding='utf-8') as f:
        #     json.dump(multi_abs,f,indent=2)

        return heading