
coverage_prompt = '''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated literature review to a human-written literature review on the topic of [TOPIC]. 

Human-Written Literature Review (Gold Standard): 
[GROUND TRUTH REVIEW] 

Generated Literature Review (To be evaluated): 
[GENERATED REVIEW] 

**Evaluation Requirements**: The human-written literature review serves as the gold standard. Your job is to assess how well the generated literature review compares in terms of coverage. Carefully analyze both reviews and provide a score. Evaluate Coverage (Score out of 100). Assess how comprehensively the generated review covers the content from the human-written review. Consider: 

• The percentage of key subtopics addressed from the human-written review. 
• The depth of discussion for each subtopic compared to the human-written version. 
• Balance between different areas within the topic as presented in the human-written review. 

Only return only a numerical score out of 100, where 100 represents perfect alignment with the human-written literature review, without providing any additional information.
'''

relevance_prompt = '''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated literature review to a human-written literature review on the topic of [TOPIC].

Human-Written Literature Review (Gold Standard):
[GROUND TRUTH REVIEW]

Generated Literature Review (To be evaluated):
[GENERATED REVIEW]

**Evaluation Requirements**: The human-written literature review serves as the gold standard. Your job is to assess how well the generated literature review compares in terms of relevance. Carefully analyze both reviews and provide a score.

**Evaluate Relevance (Score out of 100)**. Evaluate how well the generated literature review aligns with the focus and content of the human-written literature review. Consider:

• Alignment with the core aspects of [TOPIC] as presented in the human-written literature review.
• Relevance of examples and case studies compared to those in the human-written literature review.
• Appropriateness for the target audience as demonstrated by the human-written literature review.
• Exclusion of tangential or unnecessary information not present in the human-written version.

Only return only a numerical score out of 100, where 100 represents perfect alignment with the human-written literature review, without providing any additional information.

'''

structure_prompt = '''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated literature review to a human-written literature review on the topic of [TOPIC].

Human-Written Literature Review (Gold Standard):
[GROUND TRUTH REVIEW]

Generated Literature Review (To be evaluated):
[GENERATED REVIEW]

**Evaluation Requirements**: The human-written literature review serves as the gold standard. Your job is to assess how well the generated literature review compares in terms of structure. Carefully analyze both reviews and provide a score.

**Evaluate Structure (Score out of 100)**. Assess how well the generated literature review’s organization and flow match that of the human-written literature review. Consider:

• Similarity in logical progression of ideas.
• Presence of a clear hierarchy of sections and subsections comparable to the humanwritten literature review.
• Appropriate use of headings and subheadings in line with the human-written version.
• Overall coherence within and between sections relative to the human-written literature review.

Only return only a numerical score out of 100, where 100 represents perfect alignment with the human-written literature review, without providing any additional information.
'''

adequacy_prompt = '''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated literature review to a human-written literature review on the topic of [TOPIC].

Human-Written Literature Review (Gold Standard):
[GROUND TRUTH REVIEW]

Generated Literature Review (To be evaluated):
[GENERATED REVIEW]

**Evaluation Requirements**: The human-written taxonomy serves as the gold standard. Your job is to assess whether the generated taxonomy is suitable for learning this field. This means evaluating whether the taxonomy is practically usable and meets the basic standards for organizing and understanding the key areas of the domain.

**Evaluate Adequacy (Answer with Yes or No)**. Consider:

• Does the taxonomy clearly cover the main components of the field?
• Is the structure understandable and helpful for someone new to the topic?
• Would this taxonomy assist a reader in gaining a coherent overview of the research area?

Only return “Yes” if the taxonomy is sufficiently clear, complete, and useful for understanding the field. Otherwise, return “No”. Return only a single word: Yes or No, without any additional explanation.
'''

validity_prompt='''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated literature review to a human-written literature review on the topic of [TOPIC].

Human-Written Literature Review (Gold Standard):
[GROUND TRUTH REVIEW]

Generated Literature Review (To be evaluated):
[GENERATED REVIEW]

**Evaluation Requirements**: The human-written taxonomy serves as the gold standard. Your job is to assess how accurately the generated taxonomy reflects factual information and represents the domain’s conceptual structure.

**Evaluate Validity (Score from 1 to 5)**. Consider:

• Are the concepts and categories factually correct?
• Does the taxonomy accurately represent the key components and structure of the domain?
• Are there significant omissions, misclassifications, or conceptual errors?

Use the following scoring scale:
• 1 – Completely inaccurate, with significant factual errors or misrepresentations of the domain.
• 2 – Mostly inaccurate, capturing only a few correct facts but failing to represent the domain coherently.
• 3 – Moderately accurate, containing some factual correctness but missing important concepts or relationships.
• 4 – Mostly accurate, representing the domain well with minor factual inaccuracies or omissions.
• 5 – Highly accurate, thoroughly reflecting the domain’s factual structure with no noticeable errors.

Only return a **single number between 1 and 5**, without providing any additional explanation.
'''

usefulness_prompt = '''
**Instruction**: You are an expert in literature review evaluation, tasked with comparing a generated taxonomy to a human-written taxonomy on the topic of [TOPIC].

Human-Written Literature Review (Gold Standard):
[GROUND TRUTH REVIEW]

Generated Literature Review (To be evaluated):
[GENERATED REVIEW]

**Evaluation Requirements**: The human-written taxonomy serves as the gold standard. Your job is to assess how **useful** the generated taxonomy would be for someone trying to learn about this field. This includes its clarity, coverage, and overall helpfulness in understanding the research landscape.

**Evaluate Usefulness (Score from 1 to 5)**. Consider:
• Does the taxonomy identify relevant and meaningful aspects of the field?
• Would it help a reader gain an overview of the domain and understand how research areas relate?
• Is it presented in a structured and comprehensible way?

Please rate usefulness on a scale from **1 to 5**, where:
**1** = Useless — does not help understanding at all
**2** = Slightly useful — covers very little or is confusing
**3** = Useful — provides basic guidance but lacks completeness or clarity
**4** = Quite useful — generally helpful with minor gaps
**5** = Fully useful and adequate — clearly structured and suitable for learning the field

Only return a single number from 1 to 5, without any additional explanation.
'''

not_applicable_prompt = '''
You are a research paper analysis evaluator. Your task is to assess whether a model-generated label of "Not applicable" for a specific analytical dimension is justified based on the given paper content.

Input:

A paper’s title, abstract, and optionally its introduction

A predefined analytical dimension (e.g., "Evaluation Setting", "Research Focus → Hallucination Detection")


Your goals:

Evaluate whether the "Not applicable" label is accurate:

Read the provided paper content

Determine whether there is any information in the text relevant to the given dimension

If any relevant information exists, the label "Not applicable" is incorrect.

Only output your binary judgment: "Correct" or "Incorrect"

Paper:
[PAPER]

Dimension:
[TOPIC]
'''


not_summary_prompt = '''
You are an evaluator tasked with determining whether the label "Not applicable" should be assigned to a specific analytical dimension based on a paper's content.

Input:

A paper’s title, abstract, and optionally its introduction

A specific analytical dimension (e.g., "Evaluation Setting", "Research Focus → Hallucination Detection")

Your task:

Assess whether the paper provides any information relevant to the given dimension:

If any relevant content is found, then "Not applicable" should not be used.

If no relevant content is found, then "Not applicable" is a justified label.

Only output your binary decision:
Yes (if "Not applicable" is appropriate)
No (if the label is inappropriate)


Paper:
[PAPER]

Dimension:
[TOPIC]
'''

'''
Adequacy is a binary metric where evaluators respond with “Yes” or “No” to the question: “Compared to the taxonomy written by humans, is this taxonomy suitable for learning this field?” This assesses whether the taxonomy is practically usable and meets the fundamental requirements for understanding the domain.

Validity, on the other hand, is rated on a scale from 1 to 5, evaluating the degree to which the taxonomy accurately reflects factual information and represents the domain’s conceptual structure. The scoring is defined as follows:

• 1 – Completely inaccurate, with significant factual errors or
misrepresentations of the domain.
• 2 – Mostly inaccurate, capturing only a few correct facts but
failing to represent the domain coherently.
• 3 – Moderately accurate, containing some factual correctness
but missing important concepts or relationships.
• 4 – Mostly accurate, representing the domain well with minor
factual inaccuracies or omissions.
• 5 – Highly accurate, thoroughly reflecting the domain’s factual
structure with no noticeable errors.
'''

