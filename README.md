<h1 align="center">Forking</h1>
<h3 align="center">Sudden Overfitting Under Replay</h3>

<p align="center">When the same text returns, fitting and generalization can take different paths.</p>

<p align="center">
  <a href="https://guoshaoyang-pku.github.io/forking/"><img src="https://img.shields.io/badge/Website-Project_page-245D80?style=flat-square" alt="Website"></a>
  <a href="#paper"><img src="https://img.shields.io/badge/Paper-Coming_soon-64748B?style=flat-square" alt="Paper coming soon"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX_soon-64748B?style=flat-square" alt="Citation coming soon"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2F7565?style=flat-square" alt="MIT License"></a>
</p>

Local n-gram memory can improve training fit while held-out predictions deteriorate under replay. **Forking** studies this divergence through repeated-pass controls, continuation coverage, and interventions on the memory pathway.

![Local memory and diverging training and validation trajectories.](results/figures/overview.png)

## Inspect the next-token predictions

[![Figure 8: Local sharpening and cross-context recovery.](results/figures/mechanism.png)](results/figures/mechanism.pdf)

**Figure 8:** Local sharpening and cross-context recovery, illustrated schematically. Dashed marks show prior heights; darker extensions show gains. Others denotes the mean probability of the top 20 remaining tokens. Measured distributions for *there is a* appear in Figure 21(e,f).

[Measured distributions](results/figures/context-examples.pdf) · [Experimental results](results/)

## Reproduce

Python 3.10 or newer; the numerical checks use only the standard library.

~~~sh
git clone --branch release --single-branch https://github.com/guoshaoyang-pku/forking.git
cd forking
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py
~~~

Recomputes **eight coverage fits across three seeds** and checks **72 context endpoints** from the bundled observations.

[Analysis and plotting](reproduce/README.md) · [Training the baseline and no-gram control](reproduce/TRAINING.md) · [Configurations](configs/)

Training requires external tokenized inputs and a frequency index; data, tokenizer, and model weights are not bundled.

## Paper

*Forking: Sudden Overfitting Under Replay* — paper link to be added.

## Citation

BibTeX will be added with the paper.

## License

[MIT](LICENSE).
