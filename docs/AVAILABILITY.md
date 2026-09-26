# Materials and availability

The release branch contains the core implementation, baseline configurations, compact numerical evidence, and project page. Development records and history remain on the existing development branch. The release branch can be cloned independently:

~~~sh
git clone --branch release --single-branch https://github.com/guoshaoyang-pku/forking.git
~~~

| Material | Current availability |
| --- | --- |
| Core training and measurement code | Included; original source identities recorded in the release manifest |
| Coverage numerical analysis | Included and executable with Python's standard library |
| Context-level endpoint analysis | Included as CSVs with checkpoint identities and export semantics |
| Paper figures | Ten pinned vector PDFs, web renders, and provenance notes |
| Training inputs | Byte identities and split are included; token shards, tokenizer asset, frequency index, and public acquisition instructions are not bundled |
| Model checkpoints | Identities recorded for context examples; weight files are not bundled |
| Fresh GPU training reproduction | Not performed as part of release packaging |
| CPU validation | Numerical checks and 15 implementation/launcher tests passed locally; checks also passed after cloning the published branch |
| GitHub Actions | Workflow configured; hosted verification has not run because GitHub blocked runner startup for the account |
| Final manuscript PDF and citation | Final release artifact and author-approved citation metadata still need to be attached |
| License | A project license has not yet been selected; no open-source license is implied |

The figure files were copied from paper revision 83f17dbc441e61d6a73b7cc0952801f6fc4f3999's working tree and pinned individually by SHA-256. Concurrent manuscript edits are not represented as a clean final source revision. An older compiled PDF is deliberately not presented as the final manuscript.

Before promoting this branch to the final release, attach the approved final paper/citation and license, publish the precise data/tokenizer acquisition instructions, and validate the training command on those assets. The included workflow is configured to run the CPU analysis checks and launcher tests when hosted runners are available.
