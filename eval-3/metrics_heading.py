import re
from typing import List, Optional
import os
from tqdm import tqdm
from flair.data import Sentence
from flair.nn import Classifier
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
tagger = Classifier.load('ner')

encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')



def load_str(path):
    with open(path, 'r') as f:
        if "autosurvey" in path.lower():
            dec_lines = [d[1:] for d in f.readlines()]
            return ''.join(dec_lines)
        else:
            return ''.join(f.readlines())

def get_sections(path):

    s = load_str(path)
    # print(s)
    # s = re.sub(r"\d+\.\ ", '#', s)
    # print(s)
    sections = []
    for line in s.split('\n'):
        line = line.strip()
        # print("1111",line)
        # if "# References" in line:
        #     break
        if line.startswith('#'):
            if len(line.split("#"))>4:
                continue
            if any(line.strip("#").strip().lower().startswith(keyword) for keyword in ["references", "external links", "see also", "notes"]):
                break
            sections.append(line.strip('#').strip())
    # print(sections)
    return sections


def card(l):
    encoded_l = encoder.encode(list(l))
    cosine_sim = cosine_similarity(encoded_l)
    # print(cosine_sim.shape)
    # print(cosine_sim.sum(axis=1))
    soft_count = 1 / cosine_sim.sum(axis=1)
    # print(len(soft_count))
    # print(soft_count)
    # print(soft_count.mean(), soft_count.sum())
    # assert cosine_sim.mean(axis=1) == cosine_sim.sum(axis=1)
    # assert soft_count.mean()==soft_count.sum()
    return soft_count.sum()


def card_mean(l):
    encoded_l = encoder.encode(list(l))
    cosine_sim = cosine_similarity(encoded_l)
    soft_count = 1 / cosine_sim.sum(axis=1)

    return soft_count.mean()

def heading_soft_recall(golden_headings: List[str], predicted_headings: List[str]):
    """
    Given golden headings and predicted headings, compute soft recall.
        -  golden_headings: list of strings
        -  predicted_headings: list of strings

    Ref: https://www.sciencedirect.com/science/article/pii/S0167865523000296
    """

    g = set(golden_headings)
    p = set(predicted_headings)


    if len(p) == 0:
        return 0
    
    card_p = card(p)
    card_pug = card(g.union(p))
    card_g = card(g)
    # print(card_p / card_g , card_pug / card_g)
    # print(card_p , card_pug , card_g)

    
    card_p_mean = card_mean(p)
    card_pug_mean = card_mean(g.union(p))
    card_g_mean = card_mean(g)
    # print(card_p_mean / card_g_mean , card_pug_mean / card_g_mean)
    # print(card_p_mean , card_pug_mean, card_g_mean)
    # card_intersection = card_g + card_p - card(g.union(p))
    # return card_intersection / card_g
    # print(g.union(p))
    # print(card(g.union(p)))
    # print(card_p, card_pug, card_g)
    # print(111)
    # return (card_p_mean / card_g_mean , card_pug_mean / card_g_mean)
    return (card_p/card_g, card_pug/card_g)
    # card_intersection = 1 + card_p / card_g - card(g.union(p)) / card_g


def extract_entities_from_list(l):
    entities = []
    for sent in l:
        if len(sent) == 0:
            continue
        sent = Sentence(sent)
        tagger.predict(sent)
        entities.extend([e.text for e in sent.get_spans('ner')])

    entities = list(set([e.lower() for e in entities]))

    return entities


def heading_entity_recall(golden_entities: Optional[List[str]] = None,
                          predicted_entities: Optional[List[str]] = None,
                          golden_headings: Optional[List[str]] = None,
                          predicted_headings: Optional[List[str]] = None):
    """
    Given golden entities and predicted entities, compute entity recall.
        -  golden_entities: list of strings or None; if None, extract from golden_headings
        -  predicted_entities: list of strings or None; if None, extract from predicted_headings
        -  golden_headings: list of strings or None
        -  predicted_headings: list of strings or None
    """
    if golden_entities is None:
        assert golden_headings is not None, "golden_headings and golden_entities cannot both be None."
        golden_entities = extract_entities_from_list(golden_headings)
    if predicted_entities is None:
        assert predicted_headings is not None, "predicted_headings and predicted_entities cannot both be None."
        predicted_entities = extract_entities_from_list(predicted_headings)
    # print(golden_entities, predicted_entities)
    g = set(golden_entities)
    p = set(predicted_entities)
    # print(g)
    # print(p)
    if len(g) == 0:
        return 0
    else:
        return len(g.intersection(p)) / len(g)


def article_entity_recall(golden_entities: Optional[List[str]] = None,
                          predicted_entities: Optional[List[str]] = None,
                          golden_article: Optional[str] = None,
                          predicted_article: Optional[str] = None):
    """
    Given golden entities and predicted entities, compute entity recall.
        -  golden_entities: list of strings or None; if None, extract from golden_article
        -  predicted_entities: list of strings or None; if None, extract from predicted_article
        -  golden_article: string or None
        -  predicted_article: string or None
    """
    if golden_entities is None:
        assert golden_article is not None, "golden_article and golden_entities cannot both be None."
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', golden_article)
        golden_entities = extract_entities_from_list(sentences)
    if predicted_entities is None:
        assert predicted_article is not None, "predicted_article and predicted_entities cannot both be None."
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', predicted_article)
        predicted_entities = extract_entities_from_list(sentences)
    g = set(golden_entities)
    p = set(predicted_entities)
    if len(g) == 0:
        return 1
    else:
        return len(g.intersection(p)) / len(g)


def compute_rouge_scores(golden_answer: str, predicted_answer: str):
    """
    Compute rouge score for given output and golden answer to compare text overlap.
        - golden_answer: plain text of golden answer
        - predicted_answer: plain text of predicted answer
    """

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(golden_answer, predicted_answer)
    score_dict = {}
    for metric, metric_score in scores.items():
        score_dict[f'{metric.upper()}_precision'] = metric_score.precision
        score_dict[f'{metric.upper()}_recall'] = metric_score.recall
        score_dict[f'{metric.upper()}_f1'] = metric_score.fmeasure
    return score_dict


def call_soft_heading(pred_paths, ground_paths):
    final_res = {}

    entity_recalls = []
    heading_soft_recalls = []
    p_path_l = []
    g_path_l = []
    heading_soft_recalls_p = []
    heading_soft_recalls_pug = []
    pred_count = []
    gold_count = []

    for pred_path, ground_path in zip(pred_paths, ground_paths):
        if os.path.exists(pred_path) and os.path.exists(ground_path):        
            pred_sections = get_sections(pred_path)    
            if len(pred_sections) == 0:
                print("no predictions:",pred_path)
                continue 
            gt_sections = get_sections(ground_path)
            # print("gt_sections",gt_sections)
            # print("pred_sections", pred_sections)
            # print(heading_entity_recall(golden_headings=gt_sections, predicted_headings=pred_sections))


            entity_recalls.append(heading_entity_recall(golden_headings=gt_sections, predicted_headings=pred_sections))
                # heading_soft_recalls.append(heading_soft_recall(gt_sections, pred_sections))
            try:
                card_p, card_pug = heading_soft_recall(gt_sections, pred_sections)
                heading_soft_recalls_pug.append(card_pug)
                heading_soft_recalls_p.append(card_p)
                heading_soft_recalls.append(1+card_p-card_pug)  
                p_path_l.append(pred_path)    
                g_path_l.append(ground_path) 
                pred_count.append(len(pred_sections))
                gold_count.append(len(gt_sections))
            except Exception as e:
                print(e)
                # heading_soft_recalls.append(0)
                # print("gt_sections",gt_sections)
                # print("pred_sections", pred_sections)
                print("no predictions:",pred_path)
    
    micro_count = [pred_count[i]/gold_count[i] for i in range(len(pred_count))]

    final_res = {
        "Average Prediction Items": (np.mean(pred_count), np.std(pred_count)),
        "Average GroundTruth Items": (np.mean(gold_count), np.std(gold_count)),
        "Average Items Micro": (np.mean(micro_count), np.std(micro_count)),
        "Average Items Macro": (np.mean(pred_count)/np.mean(gold_count), 0),        
        "Average Entity Recall": (np.mean(entity_recalls), np.std(entity_recalls)),
        "Average Heading Soft Recall P": (np.mean(heading_soft_recalls_p), np.std(heading_soft_recalls_p)),
        "Average Heading Soft Recall P union G": (np.mean(heading_soft_recalls_pug), np.std(heading_soft_recalls_pug)),
        "Average Heading Soft Recall": (np.mean(heading_soft_recalls), np.std(heading_soft_recalls))
    }

    return final_res



def main(args):
    df = pd.read_csv(args.input_path)
    entity_recalls = []
    heading_soft_recalls = []
    topics = []
    for _, row in tqdm(df.iterrows()):
        topic_name = row['topic'].replace(' ', '_').replace('/', '_')
        gt_sections = get_sections(os.path.join(args.gt_dir, 'txt', f'{topic_name}.txt'))
        # gt_sections = get_sections(os.path.join(args.gt_dir, 'txt', f'{topic_name}.txt'))
        pred_sections = get_sections(os.path.join(args.pred_dir, topic_name, args.pred_file_name))
        # pred_sections = get_sections(os.path.join(args.pred_dir, topic_name, args.pred_file_name))
        entity_recalls.append(heading_entity_recall(golden_headings=gt_sections, predicted_headings=pred_sections))
        heading_soft_recalls.append(heading_soft_recall(gt_sections, pred_sections))
        topics.append(row['topic'])

    results = pd.DataFrame({'topic': topics, 'entity_recall': entity_recalls, 'heading_soft_recall': heading_soft_recalls})
    results.to_csv(args.result_output_path, index=False)
    avg_entity_recall = sum(entity_recalls) / len(entity_recalls)
    avg_heading_soft_recall = sum(heading_soft_recalls) / len(heading_soft_recalls)
    print(f'Average Entity Recall: {avg_entity_recall}')
    print(f'Average Heading Soft Recall: {avg_heading_soft_recall}')




if __name__=="__main_":
    ground_path_root = "/home/kunzhu/projects/taxonomy_2/eval/ground_new_outline"

    entity_recalls = []
    heading_soft_recalls = []
    p_path_l = []
    g_path_l = []
    heading_soft_recalls_p = []
    heading_soft_recalls_pug = []
    pred_count = []
    gold_count = []

    for taxo_file in tqdm(os.listdir(ground_path_root)):
        arxiv_id = taxo_file[:-3].replace("ovo","/")
        # pred_path = f"/home/kunzhu/projects/Knowledge_Navigator/Navigator/cache/{arxiv_id}/outline.md"
        pred_path = f"/home/kunzhu/projects/taxonomy_2/eval/tntllm/heading/{arxiv_id}.md"
        ground_path = f"{ground_path_root}/{arxiv_id}.md"


        if os.path.exists(pred_path) and os.path.exists(ground_path):        
            pred_sections = get_sections(pred_path)            
            gt_sections = get_sections(ground_path)
            print("gt_sections",gt_sections)
            print("pred_sections", pred_sections)
            print(heading_entity_recall(golden_headings=gt_sections, predicted_headings=pred_sections))

            entity_recalls.append(heading_entity_recall(golden_headings=gt_sections, predicted_headings=pred_sections))
            # heading_soft_recalls.append(heading_soft_recall(gt_sections, pred_sections))
            try:
                card_p, card_pug = heading_soft_recall(gt_sections, pred_sections)
                heading_soft_recalls_pug.append(card_pug)
                heading_soft_recalls_p.append(card_p)
                heading_soft_recalls.append(1+card_p-card_pug)  
                p_path_l.append(pred_path)    
                g_path_l.append(ground_path) 
                pred_count.append(len(pred_sections))
                gold_count.append(len(gt_sections))
            except:
                # heading_soft_recalls.append(0)
                print("no predictions:",pred_path)

             
        else:
            print(pred_path,os.path.exists(pred_path), ground_path, os.path.exists(ground_path))


    # print("Average Entity Recall", sum(entity_recalls) / len(entity_recalls))
    # print("Average Heading Soft Recall", sum(heading_soft_recalls) / len(heading_soft_recalls))

    # cal_ted_md(p_path_l, g_path_l, os.path.join("./","CEDS.txt"))
    with open(os.path.join("./","CEDS.txt"),'a') as f:

        print("Average Prediction Items", sum(pred_count) / len(pred_count),file=f)
        print("Average GroundTruth Items", sum(gold_count) / len(gold_count),file=f)
        micro_count = [pred_count[i]/gold_count[i] for i in range(len(pred_count))]
        print(micro_count)
        print("Average Items Micro", sum(micro_count) / len(micro_count),file=f)
        print("Average Items Macro", (sum(pred_count) / len(pred_count)) / (sum(gold_count) / len(gold_count)),file=f)        
        print("Average Entity Recall", sum(entity_recalls) / len(entity_recalls),file=f)
        print("Average Heading Soft Recall P", sum(heading_soft_recalls_p) / len(heading_soft_recalls_p),file=f)
        print("Average Heading Soft Recall P union G", sum(heading_soft_recalls_pug) / len(heading_soft_recalls_pug),file=f)
        print("Average Heading Soft Recall", sum(heading_soft_recalls) / len(heading_soft_recalls),file=f)



