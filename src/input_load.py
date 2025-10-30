import json
import requests
from threading import Thread
import time
import os
from tqdm import tqdm
import html2text
import random



class DataLoader:
    def __init__(self, ss_api, save_html_path, save_md_path) -> None:
        # self.db_path = db_path
        self.ss_api = ss_api
        self.save_html_path = save_html_path
        self.md_path = save_md_path


    # def load_papers_db():
    def __search_id_batch(self, query_list):
        # for  in data_tur:
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)  
        api_key = self.ss_api
        headers = {'x-api-key': api_key}
        try:
            response = requests.post(
                'https://api.semanticscholar.org/graph/v1/paper/batch',
                params={'fields': 'externalIds,title,abstract'},
                headers=headers,
                json={"ids": query_list}
            )
            # print(json.dumps(r.json(), indent=2))
            # print(response.json())
            time.sleep(0.5)

            return response.json()
            # with open(os.path.join(save_path, clean_string(query) + '.json'), "wb") as code:
            #     code.write(response)
        except Exception as e:
            print(e)

        return None



    def __sample_intro_files(self, loaded_id_set, n):
        # 获取所有以.md结尾的文件
        folder_path = self.md_path
        md_files = [f for f in os.listdir(folder_path) if f.endswith('.intro') and os.path.isfile(os.path.join(folder_path, f)) and f.strip(".intro") not in loaded_id_set]
        
        # 如果文件数不足n，则取全部
        sample_count = min(n, len(md_files))

        if sample_count < 1:
            return []
        
        # 随机抽取sample_count个文件
        sampled_files = random.sample(md_files, sample_count)
        sampled_ids = [f.strip(".intro").replace("ovo","/") for f in sampled_files]
        sampled_papers = []
        for filename,paperid in zip(sampled_files,sampled_ids):
            intro_path = os.path.join(folder_path, filename)
            with open(intro_path, 'r', encoding='utf-8') as f:
                intro_text = f.readlines()
            sampled_papers.append({
                "arxiv_id":paperid,
                "introduction": '\n'.join(intro_text)
            })

        sampled_aids = ["ArXiv:"+pid for pid in sampled_ids]

        sampled_paper_info = self.__search_id_batch(sampled_aids)
        # print(len(sampled_paper_info), len(sampled_papers))
        assert len(sampled_paper_info) == len(sampled_papers)
        
        for paper_meta, paper_info in zip(sampled_paper_info,sampled_papers):
            if paper_meta is not None:
                paper_info["paperId"]=paper_meta["paperId"]
                paper_info["title"]=paper_meta["title"]
                paper_info["abstract"]=paper_meta["abstract"]

        # 返回
        return sampled_papers



    def load_survey_papers(self, data_path):

        # survey = json.load(open(self.db_path,'r',encoding='utf-8'))
        survey_info = json.load(open(data_path,'r',encoding='utf-8'))
        papers_info = survey_info["papers"]

        # papers = [{"paperId":p["paperId"],"title":p["title"],"abstract":p["abstract"],} for p in papers]
        
        paper_ssid_aid = {}
        aid_set = set()
        for index, paperInfo in papers_info.items():
            ss_id = paperInfo["paperId"]
            if paperInfo is not None and "externalIds" in paperInfo and "ArXiv" in paperInfo["externalIds"]:
                paper_ssid_aid[ss_id] = paperInfo["externalIds"]["ArXiv"]   
                aid_set.add(paperInfo["externalIds"]["ArXiv"])     

        papers = []
        paper_index = []

        for index, p in papers_info.items():
            arxiv_id = paper_ssid_aid.get(p["paperId"],None)


            papers.append({
                "paperId":p["paperId"], 
                "arxiv_id":arxiv_id,
                "title":p["title"],
                "abstract":p["abstract"] 
                })
            paper_index.append(int(index))

        return survey_info['title'], papers, paper_index
    
    def load_survey_papers_noise(self, data_path, ratio=0.1):

        # survey = json.load(open(self.db_path,'r',encoding='utf-8'))
        survey_info = json.load(open(data_path,'r',encoding='utf-8'))
        papers_info = survey_info["papers"]

        # papers = [{"paperId":p["paperId"],"title":p["title"],"abstract":p["abstract"],} for p in papers]
        
        paper_ssid_aid = {}
        aid_set = set()
        for index, paperInfo in papers_info.items():
            ss_id = paperInfo["paperId"]
            if paperInfo is not None and "externalIds" in paperInfo and "ArXiv" in paperInfo["externalIds"]:
                paper_ssid_aid[ss_id] = paperInfo["externalIds"]["ArXiv"]   
                aid_set.add(paperInfo["externalIds"]["ArXiv"])     

        
        papers = []
        paper_index = []

        for index, p in papers_info.items():
            arxiv_id = paper_ssid_aid.get(p["paperId"],None)

            papers.append({
                "paperId":p["paperId"], 
                "arxiv_id":arxiv_id,
                "title":p["title"],
                "abstract":p["abstract"]
                })
            paper_index.append(int(index))

        if ratio > 0:
            noise_papers = self.__sample_intro_files(aid_set, int(len(papers)*ratio))
            print(len(papers),len(noise_papers))
            for paper in noise_papers:
                papers.append(paper)
                paper_index.append(len(paper_index))


        return survey_info['title'], papers, paper_index
