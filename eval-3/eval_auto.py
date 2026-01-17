"""Compute outline quality metrics on a dataset.

The script expects
    - a CSV file (args.input_path) with a column 'topic' containing the topics for evaluation.
    - a directory (args.gt_dir) containing human-written articles. The articles should be named as txt/{topic_name}.txt.
    - a directory (args.pred_dir) containing generated outlines. The outlines should be named as {topic_name}/{args.pred_file_name}.
"""

import os.path
# import re
from argparse import ArgumentParser
# import pandas as pd
# from tqdm import tqdm
from metrics_heading import call_soft_heading
from metric_ceds import cal_ted_md
from metric_cluster import cal_clustering_all
# import json
# import requests



def evaluations(pred_path,ground_root,save_score):

    # pred_path = "./res/test/heading"
    # ground_root = "./ground_outline"
    check_file = ['2404.00699.md', '2404.18231.md', '2410.03751.md', '2210.12714.md', '2409.18786.md', '2406.17624.md', '2312.11562.md', '2203.05227.md', '2407.18369.md', '2302.01859.md', '2405.17935.md', '2106.15561.md', '2212.09597.md', '2406.11289.md', '2312.07913.md', '2312.01700.md', '2203.01054.md', '2405.18653.md', '2401.01313.md', '2408.06361.md', '2212.13465.md', '2406.01171.md', '2312.03863.md', '2403.18105.md', '2410.22180.md']
    # save_score = os.path.join("./","a-Evaluation_Scores-noise.txt")

    p_path_l = []
    g_path_l = []

    if not os.path.exists(pred_path):
        print("No Predict Path:", pred_path)
        return
    else:
        print("Predict Path:", pred_path)

    for pred_name in os.listdir(pred_path):
        if pred_name not in check_file:
            continue
        pred_file = os.path.join(pred_path, pred_name)
        ground_file = os.path.join(ground_root, pred_name)
        # pred_file = os.path.join(pred_path, pred_name[:-3],"heading.md")
        # print(pred_file)
        if os.path.exists(pred_file) and os.path.exists(ground_file):
            p_path_l.append(pred_file)
            g_path_l.append(ground_file)

    print("Check Files:", len(p_path_l))
    if len(p_path_l) < 1:
        return

    score = {}

    score.update(cal_ted_md(p_path_l, g_path_l))

    score.update(call_soft_heading(p_path_l, g_path_l))

    c_score, len_nmi = cal_clustering_all(p_path_l, g_path_l)
    score.update(c_score)

    with open(save_score, 'a') as f:
        # f.write(pred_path+"\n")
        print(pred_path, len(p_path_l), file=f)
        for metric, (mean, std) in score.items():
            print(f"{metric}: {mean:.4f} ± {std:.4f}", file=f)

        list_final = [
            round(score["NMI"][0]*100, 2),
            round(score["ARI"][0]*100, 2),
            round(score["Purity"][0]*100, 2),
            round(score["ced_score"][0], 2),
            round(score["Average Heading Soft Recall"][0]*100, 2),
            round(score["Average Items Micro"][0], 2),
        ]
        print("\t".join([str(x) for x in list_final]), file=f)

        f.write("-"*50+"\n")




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default="./ground_outline", help='Path of human-written articles.')
    parser.add_argument('--pred_dir', type=str, default="./res/test/heading", help='Path of generated outlines.')
    parser.add_argument('--output_path', type=str, default="Evaluation_Scores.txt", help='Path to save the results.')

    args = parser.parse_args()
    print(args.pred_dir, args.gt_dir, args.result_output_path)
    evaluations(args.pred_dir, args.gt_dir, args.output_path)

