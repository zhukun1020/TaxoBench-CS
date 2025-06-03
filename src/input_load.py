import json
import requests
from threading import Thread
import time
import os
from tqdm import tqdm
import html2text

from bs4 import BeautifulSoup


class DataLoader:
    def __init__(self, ss_api, save_html_path, save_md_path) -> None:
        # self.db_path = db_path
        self.ss_api = ss_api
        self.save_html_path = save_html_path
        self.md_path = save_md_path
        self.arxiv_ratio_list = []


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


    def __load_aid_from_ssid(self,paper_ssid_index):
        ssid_list = list(paper_ssid_index.keys())
        papers = self.__search_id_batch(ssid_list)
        assert len(papers) == len(ssid_list)
        aid_dic = {}
        for paperInfo,ssid in zip(papers,ssid_list):
                # print(paperInfo)
                if paperInfo is not None and "externalIds" in paperInfo and "ArXiv" in paperInfo["externalIds"]:
                    aid_dic[paperInfo["externalIds"]["ArXiv"]] = (ssid, paper_ssid_index[ssid])
        return aid_dic
    

    def __check_html_download(self, html_path):

        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        abstract_div = soup.find('div', class_='ltx_abstract')

        return abstract_div is not None


    def __craw_html(self, save_html_path, id_list, thread_num=10, max_num=10000):

        if not os.path.exists(save_html_path):
            os.makedirs(save_html_path)
        papers = []
        have_done = os.listdir(save_html_path)
        for id in id_list:
            file_name = id.replace("/","ovo").strip() + '.html'
            if file_name not in have_done:
                papers.append(id.strip())
            elif not self.__check_html_download(os.path.join(save_html_path,file_name)):
                papers.append(id.strip())

        print(len(papers), "papers need to craw")
        if len(papers) > 0:
            threads = []
            papers = papers[:min(len(papers),max_num)]
            batch = int(len(papers)/thread_num) + 1
            for i in range(thread_num):
                threads.append(Thread(target=self.__run, args=(papers[i*batch: (i+1)*batch], save_html_path)))
            for idx, t in enumerate(threads):
                # print(idx, ' start')
                t.start()
            for t in threads:
                t.join()


    def __run(self, paper_list,savepath):
        for id in tqdm(paper_list):
            # url = paper[:-4]
            try:
                text = requests.get("https://ar5iv.labs.arxiv.org/html/" + id).text
                # text = requests.get("https://arxiv.org/html/" + url, verify=False).text
                with open(os.path.join(savepath,id.replace("/","ovo") + '.html'), 'w', encoding='utf-8') as fp:
                    fp.write(text)
                    time.sleep(2)
            except:
                continue


    def __html_to_markdown(self, html_path, markdown_path):
        if not os.path.exists(html_path):
            return

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        markdown = html2text.HTML2Text()
        markdown.ignore_links = False  # 如果你希望保留链接
        markdown_text = markdown.handle(html_content)

        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        return markdown_text


    def __extract_introduction_from_html(self, html_path, intro_path):
        if os.path.exists(intro_path):
            with open(intro_path, 'r', encoding='utf-8') as f:
                intro_text = f.readlines()
            return intro_text
        else:
            return ""

        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # 找到id为S1的section，也就是Introduction部分
        intro_section = soup.find('section', id='S1')
        if not intro_section:
            return "Introduction section not found."

        # 提取文本内容（包括标题）
        intro_text = []

        # 获取标题
        title_tag = intro_section.find('h2')
        if title_tag:
            intro_text.append(f"# {title_tag.get_text(strip=True)}\n")

        # 获取正文内容
        for p in intro_section.find_all(['p', 'li']):
            intro_text.append(p.get_text(strip=True))

        with open(intro_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(intro_text))

        return '\n'.join(intro_text)


    # 示例使用
    # html_to_markdown('2105.11644.html', '2105.11644.md')

    def __craw_paper_content(self, id_list):
        # save_html_path = "../data/arxiv_html/"
        # save_md_path = "../data/arxiv_md/"
        save_html_path = self.save_html_path
        save_md_path = self.md_path
        if not os.path.exists(save_md_path):
            os.makedirs(save_md_path)
        # self.__craw_html(save_html_path, id_list)
        papers = {}
        for id in id_list:
            html_path = os.path.join(save_html_path, id.replace("/","ovo") + '.html')
            
            # markdown_path = os.path.join(save_md_path, id.replace("/","ovo") + '.md')
            # markdown_text = self.__html_to_markdown(html_path, markdown_path)

            intro_path = os.path.join(save_md_path, id.replace("/","ovo") + '.intro')
            intro_text = self.__extract_introduction_from_html(html_path, intro_path) #introduction_text
            if len(intro_text) > 0:
                papers[id] = intro_text
        return papers

#  db_path = f"/home/kunzhu/projects/AutoSurvey-main/database/survey_db/{arxiv_id}/paper_info_db.json"
    def load_survey_papers_from_dbpath(self):

        survey = json.load(open(self.db_path,'r',encoding='utf-8'))
        papers_info = survey["paper_info"].values()
        # papers = [{"paperId":p["paperId"],"title":p["title"],"abstract":p["abstract"],} for p in papers]
        
        index_path2 =self.db_path.replace("paper_info_db","paper_aid_to_index")
        if os.path.exists(index_path2):
            with open(index_path2,'r',encoding='utf-8') as f:
                paper_aid_index = json.load(f)
        else:
            index_path =self.db_path.replace("paper_info_db","paperid_to_index")
            paper_ssid_index = json.load(open(index_path))
            paper_aid_index = self.__load_aid_from_ssid(paper_ssid_index)        
            with open(index_path2,'w',encoding='utf-8') as f:
                json.dump(paper_aid_index,f)

        # print(len(paper_aid_index),len(papers_info),len(paper_aid_index)/len(papers_info))

        papers_content = self.__craw_paper_content(list(paper_aid_index.keys()))
        paper_ssid_aid = {}
        for a_id, value in paper_aid_index.items():
            # print(value)
            ss_id = value[0]
            paper_ssid_aid[ss_id] = a_id
        papers = []
        paper_index = []
        
        for index, p in survey["paper_info"].items():
            if p["paperId"] not in paper_ssid_aid:
                papers.append({
                "paperId":p["paperId"], 
                "arxiv_id":None,
                "title":p["title"],
                "abstract":p["abstract"],
                "introduction": None
                })
            else:   
                arxiv_id = paper_ssid_aid[p["paperId"]]
                papers.append({
                    "paperId":p["paperId"], 
                    "arxiv_id":arxiv_id,
                    "title":p["title"],
                    "abstract":p["abstract"],
                    "introduction": papers_content[arxiv_id]
                    })
            paper_index.append(int(index))

        return papers, paper_index


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

        papers_content = self.__craw_paper_content(list(aid_set))

        papers = []
        paper_index = []
        intro_count = 0
        
        for index, p in papers_info.items():
            arxiv_id = paper_ssid_aid.get(p["paperId"],None)
            intro_text = None
            if arxiv_id is not None and arxiv_id in papers_content:
                intro_count += 1
                intro_text = papers_content[arxiv_id]

            papers.append({
                "paperId":p["paperId"], 
                "arxiv_id":arxiv_id,
                "title":p["title"],
                "abstract":p["abstract"],
                "introduction": intro_text
                })
            paper_index.append(int(index))

        self.arxiv_ratio_list.append((len(papers), intro_count, intro_count/len(papers)))

        return survey_info['title'], papers, paper_index