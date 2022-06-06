# Fully Non-autoregressive Neural Machine Translation

This page mainly includes instructions for reproducing results from the following papers
* [fully non-autoregressive neural machine translation: tricks of the trade (Gu et al., 2020)](https://arxiv.org/abs/2012.15833).

## Installation
This code is based on an old version of [Fairseq](https://github.com/facebookresearch/fairseq), which has **changed significantly** after the above paper was published.
We previously found it was not easy to merge our code to the up-to-date fairseq version, so we decided to open-source with a [saperate code](/fairseq-stable).


To start with,
```bash
git clone --recursive git@github.com:shawnkx/Fully-NAT.git
```
(Optional) [ctcdecode](https://github.com/MultiPath/ctcdecode) is used for [advanced decoding](#advanced-decoding-methods) with beam-search and language model. Installation follows the following instruction
```
cd ctcdecode; pip install .
```
Then,
```
cd fairseq-stable
python setup.py build_ext --inplace
```
This code is tested on ```pytorch==1.7.1, cuda 10.2```.


## Dataset
We follow the instructions [here](https://github.com/facebookresearch/fairseq/blob/main/examples/nonautoregressive_translation/README.md).
You can also download the preprocessed datasets as follows:
Direction | Download
---|---
Ro <-> En | [download (.zip)](https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt16.ro-en.zip)
En <-> De | [download (.zip)](https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt14.en-de.zip)
Ja --> En | [download (.zip)](https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt20.ja-en.zip)

## Pre-trained Models
### Ro-En
Direction | Model | BLEU | | Direction | Model | BLEU 
---|---|---|---|---|---|---
Ro -> En | [`roen.distill.ctc`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/roen_distill_ctc.pt) | 34.07 | |  En -> Ro | [`enro.distill.ctc`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/enro_distill_ctc.pt)        | 33.41 
Ro -> En | [`roen.distill.ctc.glat`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/roen_distill_ctc_glat.pt)   | <b>34.16</b> | | En -> Ro | [`enro.distill.ctc.glat` ](https://dl.fbaipublicfiles.com/nat/fully_nat/models/enro_distill_ctc_glat.pt)   | 33.71
Ro -> En | [`roen.distill.ctc.vae`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/roen_distill_ctc_latent.pt) | 33.87 | | En -> Ro | [ `enro.distill.ctc.vae`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/enro_distill_ctc_latent.pt) | <b>33.79</b>

### En-De
Direction | Model | BLEU | | Direction | Model | BLEU 
---|---|---|---|---|---|---
De -> En | [`deen.distill.ctc`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/deen_distill_ctc.pt) | 30.46 | |  En -> De | [`ende.distill.ctc`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/ende_distill_ctc.pt)        | 26.51
De -> En | [`deen.distill.ctc.glat`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/deen_distill_ctc_glat.pt)   | <b>31.37</b> | | En -> De | [`ende.distill.ctc.glat` ](https://dl.fbaipublicfiles.com/nat/fully_nat/models/ende_distill_ctc_glat.pt)   | 27.20
De -> En | [`deen.distill.ctc.vae`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/deen_distill_ctc_latent.pt) | 31.10 | | En -> De | [ `ende.distill.ctc.vae`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/ende_distill_ctc_latent.pt) | <b>27.49</b>

### Ja-En
Direction | Model | BLEU | BLEU + 4gram-LM + beam20
---|---|---|---
Ja -> En |  [`jaen.distill.ctc.vae`](https://dl.fbaipublicfiles.com/nat/fully_nat/models/jaen_distill_ctc_latent.pt)  | 18.73 | 21.41 (Download [4-gram En-LM](https://dl.fbaipublicfiles.com/nat/fully_nat/models/jaen_distill_en_4gram.bin))


## Train a model
The following command will train a baseline NAT model with CTC loss.

```bash
python train.py ${databin_dir} \
    --fp16 \
    --left-pad-source False --left-pad-target False \
    --arch cmlm_transformer_ctc --task translation_lev \
    --noise 'full_mask' --valid-noise 'full_mask' \
    --dynamic-upsample --src-upsample 3 \
    --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-06 \
    --clip-norm 2.4 --dropout 0.3 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 10000 --lr 0.0005 --min-lr 1e-09 \
    --criterion nat_loss --predict-target 'all' --loss-type 'ctc' \
    --axe-eps --force-eps-zero \
    --label-smoothing 0.1 --weight-decay 0.01 \
    --max-tokens 4096 --update-freq 1 \
    --max-update 300000 --save-dir ${model_dir} --save-interval-updates 5000 \
    --no-epoch-checkpoints --keep-interval-updates 10 --keep-best-checkpoints 5 \
    --seed 2 --log-interval 100 --no-progress-bar \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter":0,"iter_decode_collapse_repetition":true}' \
    --eval-bleu-detok 'space' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe '@@ ' --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir ${ckpt_dir}/tensorboard/${expname} 
```

Use the ```--latent-dim 8```  flag to add the latent variables (VAEs). <br>
Use ```--glat-sampling-ratio 0.5 --poisson-mask  --replace-target-embed``` to add the GLAT training.

## Translate
For inference, 
```bash
python fairseq_cli/generate.py ${databin_dir} \
    --task translation_lev \
    --path checkpoints/checkpoint_best.pt \
    --gen-subset test \
    --axe-eps  --iter-decode-collapse-repetition --force-eps-zero \
    --left-pad-source False --left-pad-target False \
    --iter-decode-max-iter 0 --beam 1 \
    --remove-bpe --batch-size 200 \
```
In the end of the generation, we can see the tokenized BLEU score for the translation.
For datasets based on BPEs (Ro-En, En-De), we report standard tokenized BLEU, for datasets using sentencepiece (Ja-En), 
we report sacre-bleu by replacing ```--remove-bpe``` as ```--remove-bpe sentencepiece --scoring sacrebleu```.

Set ```--batch-size 1``` to compute the latency when decoding one sentence at a time.


## Advanced Decoding Methods
### CTC Beam Search
We can use CTC Beam Search (+ language model) to improve the translation quality
```bash
python fairseq_cli/generate.py ${databin_dir} \
    --task translation_lev \
    --path checkpoints/checkpoint_best.pt \
    --gen-subset test \
    --axe-eps  --iter-decode-collapse-repetition --force-eps-zero \
    --left-pad-source False --left-pad-target False \
    --iter-decode-max-iter 0 --beam 1 \
    --remove-bpe sentencepiece --scoring sacrebleu \
    --batch-size 20 \
    --model-overrides "{'use_ctc_beamsearch':True, 'ctc_bs_beam':${ctc_beam_size}, 'ctc_bs_alpha':${alpha}, 'ctc_bs_beta':${beta}, 'ctc_bs_lm_path':'${lm_path}'}"
```
For example, for [Ja-En model](#ja-en), we use ```alpha=0.4, beta=2.2, ctc_beam_size=20```<br> 
If ```${lm_path}``` is None, we apply CTC beam search without language model.


## Citation
```bibtex
@inproceedings{gu-kong-2021-fully,
    title = "Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade",
    author = "Gu, Jiatao  and
      Kong, Xiang",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.11",
    doi = "10.18653/v1/2021.findings-acl.11",
    pages = "120--133",
}
```