---
language:
- en
license: mit
size_categories:
- 10K<n<100K
---

## Data

Our training dataset consists of approximately 40,000 unique mathematics problem-answer pairs compiled from:

- AIME (American Invitational Mathematics Examination) problems (1984-2023)
- AMC (American Mathematics Competition) problems (prior to 2023)
- Omni-MATH dataset
- Still dataset

## Format

Each row in the JSON dataset contains:

- **problem**: The mathematical question text, formatted with LaTeX notation.
- **solution**: Offical solution to the problem, including LaTeX formatting and boxed final answers. If there is no solution, the `solution` field is an empty string.
- **answer**: The answer to the problem, usually extracted from the solution.

## Example

```json
{
  "problem": "Let $a_n=6^{n}+8^{n}$. Determine the remainder upon dividing $a_ {83}$ by $49$.",
  "solution": "$6^{83} + 8^{83} = (6+8)(6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82})$\n Becuase $7|(6+8)$, we only consider $6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82} \\pmod{7}$\n$6^{82}-6^{81}8+\\ldots-8^{81}6+8^{82} \\equiv (-1)^{82} - (-1)^{81}+ \\ldots - (-1)^1 + 1 = 83 \\equiv 6 \\pmod{7}$\n$6^{83} + 8^{83} \\equiv 14 \\cdot 6 \\equiv \\boxed{035} \\pmod{49}$",
  "answer": "35",
}
```

## Citation

```bibtex
@misc{deepscaler2025,
  title={DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL},
  author={Michael Luo and Sijun Tan and Justin Wong and Xiaoxiang Shi and William Tang and Manan Roongta and Colin Cai and Jeffrey Luo and Tianjun Zhang and Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2}},
  note={Notion Blog}
  year={2025}
}
```
