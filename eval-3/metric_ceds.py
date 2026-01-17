from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertConfig, BertModel, BertTokenizer
# https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances/-/tree/master
import edist.ted as ted
import edist.sed as sed
import edist.tree_utils as tree_utils
import edist.tree_edits as tree_edits
from tqdm import tqdm
from bert_score import BERTScorer
import os
import argparse
import nltk
import numpy as np



def get_embed(cands, model, tokenizer):
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs)
    cands_embd = []
    for cand in cands:
        # print(cand)
        cand_ids = tokenizer(cand.strip(), return_tensors="pt")
        _, pooled_output, _ = model(**cand_ids)
        cands_embd.append(pooled_output.detach().numpy())
        # outputs = model(**cand_ids)
        # print(outputs)
        # last_hidden_states = outputs.last_hidden_state
        # cands_embd.append(last_hidden_states)
    return cands_embd


def cal_embed_cos(targets_emb, cands_emb):
    cos_list_all = []
    for target_embd in targets_emb:
        score_list = []
        for cand_embd in cands_emb:
            cos = cosine_similarity(target_embd,cand_embd)
            score_list.append(cos)           
        cos_list_all.append(score_list)
    return cos_list_all


def get_most_siml(cos_list_all,targets, cands):
    assert len(cos_list_all) == len(targets)
    assert len(cos_list_all[0]) == len(cands)
    final_lis = []
    for i in range(len(targets)):
        cos_lis = list(cos_list_all[i])
        comb_lis = [(cos_lis[j],cands[j]) for j in range(len(cands))]
        sort_comb_lis = sorted(comb_lis,reverse=True,key=lambda x:x[0])
        final_lis.append(sort_comb_lis)
    return final_lis

def get_siml_matrix(cos_list_all,targets, cands):
    assert len(cos_list_all) == len(targets)
    assert len(cos_list_all[0]) == len(cands)
    final_lis = {}
    for i in range(len(targets)):
        for j in range(len(cands)):
            final_lis[fr"{targets[i]}-{cands[j]}"] = cos_list_all[i][j]
    return final_lis


def get_items(line):
    line_new = line.replace("<s1> ","\t").replace("<s2> ","\t").replace("<s3> ","\t").strip()
    items = line_new.split("\t")
    return items


def prepare_tree(line):
    # x_nodes = ['a', 'b', 'c', 'd', 'e']
    # x_adj   = [[1, 4], [2, 3], [], [], []]
    line_new = line.replace("<s1>","\t<s1>").replace("<s2>","\t<s2>").replace("<s3>","\t<s3>").strip()
    tmp_nodes = line_new.replace("<s1>","").replace("<s2>","").replace("<s3>","").split("\t")
    tmp_nodes = [n.strip() for n in tmp_nodes]
    nodes = ["root"]
    nodes.extend(tmp_nodes)
    tmp_t_lis = line_new.split("\t")
    tmp_t_lis = [n.strip() for n in tmp_t_lis]
    t_lis = []
    parent_lis = []
    last_s1 = -1
    last_s2 = -1
    adj = [[] for j in range(len(tmp_t_lis)+1)] # root_node-0
    for i in range(len(tmp_t_lis)):
        item = tmp_t_lis[i]
        if item.startswith("<s"):
            item = item[2:]
            if len(item) < 2 or item[1] != ">":
                continue        
            if item[0] == '1':
                parent_lis.append(-1)
                last_s1 = len(t_lis)       
            elif item[0] == '2':
                parent_lis.append(last_s1)
                last_s2 = len(t_lis)
            elif item[0] == '3':
                parent_lis.append(max(last_s1,last_s2))
            t_lis.append(item.replace("1>","").replace("2>","").replace("3>","").strip())

    child_lis = {}
    for i in range(len(parent_lis)):
        parent_index = parent_lis[i]
        child_index = i
        if parent_index not in child_lis:
            child_lis[parent_index] = []
        child_lis[parent_index].append(child_index)
    
    for i in range(-1,len(tmp_t_lis)): # -1-root
        if i in child_lis.keys():
            adj[i+1] = [k+1 for k in child_lis[i]]
    
    return nodes, adj

def prepare_tree_md(lines):
    # x_nodes = ['a', 'b', 'c', 'd', 'e']
    # x_adj   = [[1, 4], [2, 3], [], [], []]

    pattern=["acknowledgments","references","appendix"]

    lines_new = []
    for l in lines:
        flag=True
        if not l.startswith('#'):
            continue
        for p in pattern:
            if len(l.split("#"))>4 or l.lower().strip("#").strip().startswith(p):
            # if len(l.split("#"))!=4 or l.lower().strip("#").strip().startswith(p):
                flag=False
                break
        if flag and len(l.strip())>0:
            lines_new.append(l)

    tmp_nodes = [n.strip("#").strip() for n in lines_new]
    nodes = ["root"]
    nodes.extend(tmp_nodes)

    tmp_t_lis = [n.strip() for n in lines_new]

    t_lis = []
    parent_lis = []
    last_s1 = -1
    last_s2 = -1
    adj = [[] for j in range(len(tmp_t_lis)+1)] # root_node-0
    for i in range(len(tmp_t_lis)):
        item = tmp_t_lis[i]
        level = len(item.split("#"))-1
        if item.startswith("#"):                  
            if level == 1:
                parent_lis.append(-1)
                last_s1 = len(t_lis)       
            elif level == 2:
                parent_lis.append(last_s1)
                last_s2 = len(t_lis)
            elif level == 3:
                parent_lis.append(max(last_s1,last_s2))
            t_lis.append(item.strip("#").strip())

    child_lis = {}
    for i in range(len(parent_lis)):
        parent_index = parent_lis[i]
        child_index = i
        if parent_index not in child_lis:
            child_lis[parent_index] = []
        child_lis[parent_index].append(child_index)
    
    for i in range(-1,len(tmp_t_lis)): # -1-root
        if i in child_lis.keys():
            adj[i+1] = [k+1 for k in child_lis[i]]
    
    # print(nodes)
    # print(adj)
    return nodes, adj


def ceds():
    x_nodes = ['a', 'b', 'c', 'd', 'e']
    x_adj   = [[1, 4], [2, 3], [], [], []]
    y_nodes = ['a', 'c', 'd']
    y_adj   = [[1], [2], []]
    print('The tree edit distance between tree %s and tree %s is %d.' % (tree_utils.tree_to_string(x_nodes, x_adj), tree_utils.tree_to_string(y_nodes, y_adj), ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)))
    print('By contrast, the sequence edit distance on the node lists would be %d.' % (sed.standard_sed(x_nodes, y_nodes)))


def delta(x, y):
    if(x is None or y is None):
        return 1.
    alpha = 1.2
    return min(1,alpha*(1-siml_matrix[fr"{x}-{y}"]))
    # if(x in ['+', '*'] or y in ['+', '*']):
    #     if(x == y):
    #         return 0.
    #     else:
    #         # we forbid alignments of algebraic operators with
    #         # other types by assigning an infinite cost
    #         return np.inf
    # # at this point, we now that both entries are numbers
    # return abs(x - y) / max(x, y)


def cal_ted(dec, ref, out):

    with open(ref,"r") as f:
        gold_res = f.readlines()
    with open(dec,"r") as f:
        pred_res = f.readlines()

    assert len(gold_res) == len(pred_res)
    ced_stand = 0
    ced = 0
    sed_c = 0
    ced_stands = 0
    ceds = 0
    seds = 0

    score_com_ref = []
    score_com_dec = []
    score_com_index = []

    for i in tqdm(range(len(gold_res))):
    # for i in range(50,53):

        ref_item, ref_adj = prepare_tree_md(gold_res[i])        
        dec_item, dec_adj = prepare_tree_md(pred_res[i])
        new_dec_item = list(dec_item)

        #bert-score 
        cos_list_all = []
        targets = []
        ref_item_all = []
        for j in range(len(dec_item)):
            targets.extend([dec_item[j]]*len(ref_item))
            ref_item_all.extend(ref_item)
            # print(targets) 
            # print(111)
        score_com_index.append((len(score_com_ref),len(score_com_ref)+len(ref_item_all)))
        score_com_ref.extend(ref_item_all)
        score_com_dec.extend(targets)
            

    P, R, F1 = scorer.score(score_com_dec, score_com_ref)
    
    
    for i in tqdm(range(len(gold_res))):

        ref_item, ref_adj = prepare_tree_md(gold_res[i])        
        dec_item, dec_adj = prepare_tree_md(pred_res[i])
        score_F1 = F1[score_com_index[i][0]:score_com_index[i][1]]

        assert len(score_F1) == len(ref_item)*len(dec_item)
        cos_list_all = score_F1.reshape(-1,len(ref_item)).tolist()
        global siml_matrix
        siml_matrix = get_siml_matrix(cos_list_all,dec_item, ref_item)

        tree_ref_item = list(ref_item)
        tree_dec_item = list(dec_item)
        ced_stand_tmp = ted.standard_ted(tree_dec_item, dec_adj, tree_ref_item, ref_adj)
        ced_stand += ced_stand_tmp
        ced_tmp = ted.ted(tree_dec_item, dec_adj, tree_ref_item, ref_adj,delta)
        ced += ced_tmp
        # print(sed.standard_sed(tree_dec_item, tree_ref_item))
        sed_c_tmp = sed.standard_sed(tree_dec_item, tree_ref_item)
        sed_c += sed_c_tmp

        maxL = max(len(tree_ref_item),len(tree_dec_item))-1
        
        ced_stands += 100-ced_stand_tmp/maxL*100
        # print(ced_stand_tmp/maxL,ced_stands)
        ceds += 100-ced_tmp/maxL*100
        seds += 100-sed_c_tmp/maxL*100

    with open(out, 'a') as f:
        print(dec,file=f)
        print(fr"ced_stand:{ced_stand/len(gold_res)} ced_stands:{ced_stands/len(gold_res)}", file=f)
        print(fr"ced:{ced/len(gold_res)} ceds:{ceds/len(gold_res)}", file=f)
        print(fr"sed:{sed_c/len(gold_res)} seds:{seds/len(gold_res)}", file=f)


'''
['root', '1 Introduction', 'Other Surveys.', '2 Terminology', '2.1 A Categorization of Reasoning Tasks', '3 Reasoning Behavior of LLMs', '3.1 Behavior in Logical Reasoning Tasks', '3.1.1 Behavior in Deductive Reasoning Tasks', 'Mechanistic Evaluation.', '3.1.2 Behavior in Inductive Reasoning Tasks', '3.1.3 Behavior in Abductive Reasoning Tasks', '3.2 Behavior in Mathematical Reasoning Tasks', '3.3 Behavior in Causal Reasoning Tasks', 'Behavior in Counterfactual Reasoning Tasks.', 'Summary.', '4 Evaluation Methods', '4.1 Conclusion-Based Evaluation', '4.2 Rationale-Based Evaluation', '4.3 Interactive Evaluation', '4.4 Mechanistic Evaluation', '5 Discussion', '6 Conclusion']
[[1, 2, 4, 11, 16, 17], [], [], [3], [], [5, 9, 10], [6, 7, 8], [], [], [], [], [], [], [], [], [12, 13, 14, 15], [], [], [], [], [], []]
['root', 'Evaluating Reasoning Abilities in Large Language Models', 'Evaluating and Improving Reasoning in Large Language Models', 'Understanding and Evaluating Chain-of-Thought Processes', 'Evaluating and Understanding Chain-of-Thought Reasoning in Large Language Models', 'Cognitive Biases and Logical Reasoning in LLMs', 'Cognitive Biases and Deductive Reasoning in Large Language Models', 'Counterfactual and Causal Reasoning in LLMs', 'Evaluating Counterfactual and Causal Reasoning in Large Language Models', 'Enhancing Logical Reasoning in LLMs', 'Assessing and Enhancing Logical Reasoning in Large Language Models', 'Advanced Evaluation Methods for Cognitive and Reasoning Behaviors', 'Advanced Evaluation Methods for Cognitive and Reasoning Behaviors in LLMs']
[[1, 3, 5, 7, 9, 11], [2], [], [4], [], [6], [], [8], [], [10], [], [12], []]

'''
def cal_ted_md(dec_l, ref_l, out=None):

    # for dec, ref in zip(dec_l, ref_l):
    # gold_res = [open(ref).readlines() for ref in ref_l]
    # pred_res = [open(dec).readlines() for dec in dec_l]

    ref_item_l = []
    ref_adj_l = []
    dec_item_l = []
    dec_adj_l = []
    gold_res = [open(ref).readlines() for ref in ref_l]
    for i, dec in enumerate(dec_l):
        dec_lines = open(dec).readlines()
        if "autosurvey" in dec.lower():
            dec_lines = [d[1:] for d in dec_lines]

        dec_item, dec_adj = prepare_tree_md(dec_lines)
        dec_item_l.append(dec_item)
        dec_adj_l.append(dec_adj)

        ref_item, ref_adj = prepare_tree_md(gold_res[i])
        ref_item_l.append(ref_item)
        ref_adj_l.append(ref_adj)

    ced_stand = []
    ced = []
    sed_c = []
    ced_stands = []
    ceds = []
    seds = []

    score_com_ref = []
    score_com_dec = []
    score_com_index = []


    for i,ref_item, ref_adj,dec_item, dec_adj in tqdm(zip(range(len(gold_res)),ref_item_l,ref_adj_l,dec_item_l, dec_adj_l)):
    # for i in range(50,53):

        # ref_item, ref_adj = prepare_tree_md(gold_res[i])        
        # dec_item, dec_adj = prepare_tree_md(pred_res[i])
        # new_dec_item = list(dec_item)

        #bert-score 
        cos_list_all = []
        targets = []
        ref_item_all = []
        for j in range(len(dec_item)):
            targets.extend([dec_item[j]]*len(ref_item))
            ref_item_all.extend(ref_item)
            # print(targets) 
            # print(111)
        score_com_index.append((len(score_com_ref),len(score_com_ref)+len(ref_item_all)))
        score_com_ref.extend(ref_item_all)
        score_com_dec.extend(targets)
            
    # print(len(score_com_dec))
    # print(len(score_com_ref))
    P, R, F1 = scorer.score(score_com_dec, score_com_ref)
    # print(dec_l)
    # print(ref_l)
    # print(P, R, F1)
    
    
    for i,ref_item, ref_adj,dec_item, dec_adj in tqdm(zip(range(len(gold_res)),ref_item_l,ref_adj_l,dec_item_l, dec_adj_l)):

        # ref_item, ref_adj = prepare_tree_md(gold_res[i])        
        # dec_item, dec_adj = prepare_tree_md(pred_res[i])
        score_F1 = F1[score_com_index[i][0]:score_com_index[i][1]]

        assert len(score_F1) == len(ref_item)*len(dec_item)
        cos_list_all = score_F1.reshape(-1,len(ref_item)).tolist()
        global siml_matrix
        siml_matrix = get_siml_matrix(cos_list_all,dec_item, ref_item)

        tree_ref_item = list(ref_item)
        tree_dec_item = list(dec_item)
        ced_stand_tmp = ted.standard_ted(tree_dec_item, dec_adj, tree_ref_item, ref_adj)
        ced_stand.append(ced_stand_tmp)
        ced_tmp = ted.ted(tree_dec_item, dec_adj, tree_ref_item, ref_adj,delta)
        ced.append(ced_tmp)
        # print(sed.standard_sed(tree_dec_item, tree_ref_item))
        sed_c_tmp = sed.standard_sed(tree_dec_item, tree_ref_item)
        sed_c.append(sed_c_tmp)

        maxL = max(len(tree_ref_item),len(tree_dec_item))-1
        
        ced_stands.append(100-ced_stand_tmp/maxL*100)
        # print(ced_stand_tmp/maxL,ced_stands)
        ceds.append(100-ced_tmp/maxL*100)
        seds.append(100-sed_c_tmp/maxL*100)


    if out is not None:
        with open(out, 'a') as f:
            print("\n",dec_l[0],file=f)
            print(fr"ced_stand:{np.mean(ced_stand)} ced_stands:{np.mean(ced_stands)}", file=f)
            print(fr"ced:{np.mean(ced)} ceds:{np.mean(ceds)}", file=f)
            print(fr"sed:{np.mean(sed_c)} seds:{np.mean(seds)}", file=f)
    
    final_res = {
        "ced_stand": (np.mean(ced_stand), np.std(ced_stand)),
        "ced_stand_score": (np.mean(ced_stands), np.std(ced_stands)),
        "ced": (np.mean(ced), np.std(ced)),
        "ced_score": (np.mean(ceds), np.std(ceds)),        
        "sed": (np.mean(sed_c), np.std(sed_c)),
        "sed_score": (np.mean(seds), np.std(seds)),
    }
    return final_res




def cal_bertscore(ref_path, pred_txt,out):
    with open(pred_txt) as f:
        cands = [line.replace("<s1>",". ").replace("<s2>",". ").replace("<s3>",". ").strip() for line in f]

    with open(ref_path) as f:
        refs = [[line.replace("<s1>",". ").replace("<s2>",". ").replace("<s3>",". ").strip()] for line in f]

    P, R, F1 = scorer.score(cands, refs)
  
    with open(out, 'a') as f:
        print(pred_txt, file=f)
        print(f"System level P score: {P.mean():.5f}", file=f)
        print(f"System level R score: {R.mean():.5f}", file=f)
        print(f"System level F1 score: {F1.mean():.5f}", file=f)
            
        

def template_rate(path,out):
    template_lis = ['introduction','method','methods','data','dataset','datasets','conclusion','conclusions','reference','eassy',
    'purpose','definition','definitions','overview','summary','benefit','benefits','overview','challenge','challenges',
    'advantages','advantage','disadvantage','disadvantages','application','applications',"intoroduction","preliminaries","preliminary",
    "outlook","related","work","notations","notation","sources","methodology","background","future",
    "result","results","discussion","observations","observation","analysis"
    ]

    target_word_lis = []
    target_lis_total = open(path,'r').readlines()
    count_total = 0
    leng_total = 0
    for i in range(len(target_lis_total)): 
        target = target_lis_total[i]
        target = target.lower().strip().split("<s")
        tmp_target = " ".join(target)  
        target_word_lis.append(nltk.word_tokenize(tmp_target.replace("1>","").replace("2>","").replace("3>",""))) 

    count_rate = 0.0

    for words in target_word_lis:
        leng = len(words)
        if leng == 0:
            continue
        count = 0
        leng_total+=leng
        
        for word in words:
            if word in template_lis:
                count += 1
                count_total += 1
        # print(count/leng)
        count_rate += count/leng
    print(count_rate/len(target_word_lis))
    with open(out, 'a') as f:
            # print(output)
        print(file=f)
        print(fr"avg_target:{leng_total/len(target_word_lis)}", file=f)
        print(fr"avg_words:{count_total/len(target_word_lis)}", file=f)
        print(fr"template_rate:{100*count_rate/len(target_word_lis)}", file=f)


            
siml_matrix = {}           
# model_name = "allenai/scibert_scivocab_uncased"
# lang='en-sci', verbose=False, model_type=model_name
scorer = BERTScorer(lang="en-sci", rescale_with_baseline=False)

# class Sci_BERT(nn.Module):
#     def __init__(self, num_classes):
#         super(Sci_BERT, self).__init__()
#         self.lm = BertModel.from_pretrained("allenai/scibert_scivocab_cased", output_attentions = False, output_hidden_states = True, return_dict=False)
#         self.linear = nn.Linear(768, num_classes)
#     def forward(self, input, att_mask):
#         _, pooled_output, _ = self.lm(input, attention_mask = att_mask)
#         output = self.linear(pooled_output)
#         return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, default="Path_to_prediction_file")
    parser.add_argument("--ref_file", type=str, default="Path_to_ref_file") 
    parser.add_argument("--output_dir", type=str, default="Path_to_output_dir") 

    args = parser.parse_args()
    cal_ted(args.prediction_file, args.ref_file , os.path.join(args.output_dir,"CEDS.txt"))
    template_rate(args.prediction_file, os.path.join(args.output_dir,"CQE.txt"))


        





