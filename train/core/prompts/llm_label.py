from typing import List
import re

llm_label_prompt_template_for_line = """
You are given:
- a natural-language(or code for code completion task) Query
- a code snippet split into numbered lines (1>, 2>, 3>, ...)

Question: {query}
Code Context:
{code}

Answer the Question, using ONLY information provided in the Code Context. If no useful information
is provided, you MUST output "No answer". If some parts of the Context are used to answer, you
MUST cite ALL the corresponding lines. 

Use the symbols [ ] to indicate when a fact comes from a line in the context, e.g [1] for a fact from line 1. 
- For multi-line context, use [line1-line2], e.g. [12-25]). 
- For multi context, use [line1,line2,...], e.g. [1,3,5-7].

You should only answer the given question and should not provide any additional information

HINT: 
- For code, context should be wider than `the line just answer the question`, for example, if the question is about a variable in a class method function, include the function definition, class definition and where it is used.
- When you try to cite something, its better to cite the structure of the code. 
e.g. if you want to cite B1 in the code structure below:
```
1> if cond:
2>    A1
3>    A2
4> else:
5>    B1
```
, best citation will be [1,4,5], which keeps the structure of the `if-else` block while removing the unrelated A1, A2.

Now give your answer with citations:
"""


def fetch_llm_label_from_output(out: str) -> List[int]:
    """Parse citation ranges like [1], [3-5], [1,3,5-7] from LLM output."""
    try:
        citations = []
        matches = re.findall(r"\[(?:\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]", out)
        for match in matches:
            nums = match.strip("[]").strip().split(",")
            for num in nums:
                num = num.strip()
                if "-" in num:
                    start, end = map(int, num.split("-"))
                    citations.extend(range(start, end + 1))
                else:
                    citations.append(int(num))
    except Exception:
        return []
    return sorted(set(citations))
