## Introduction

A old fashion multi-label classification model embeded with [bert](https://github.com/google-research/bert) even for large dataset.

Some while ago, I wanna train a bert model to tag any given chinese text among some tags with a huge trainning dataset.
But after some searching I haven't found one for my onw case so I start create one.

## How to Use
- dependencies
  - bert-tensorflow = 1.0.1
	- tensorflow = 1.14

- implementing the `load_example` function to iterate the huge dataset chunks by chunks. eg. for a file:
```python
for (i, data) in enumerate(open("path/to/file")):
	(labels, text) = data.split("\t")
  # the target text 
	text_a = process_data_fn(text)
  # global id of the target text 
	guid = i 
  # the ground tag of the text according to `LABEL_COLUMNS`, 1 for true, 0 for false
  # labels is array of 0s, 1s, eg. [1,0,1,...,0]
	return  InputExample(tex=text_a, guid=guid, labels=labels, text_b=None)
```
- set up basic training parameter
	- `MAX_SEQ_LENGTH`: the length of `text_a` 
	- `BATCH_SIZE`: the samples take once when training, lower it to CPU/GPU saving
	- `BERT_VOCAB`: the path to vocab.txt file 
	- `BERT_INIT_CHKPNT`: the path to `bert_model.ckpt.index`
	- `BERT_CONFIG`:  the path to `bert_config.json`
	- `LABEL_COLUMNS`: array of label string, ['tag1','tag2','tag3']
	- `EPOCHES`: times to iterate the whole dataset

## Issues

	It is built upon `bert-tensorflow=1.0.1`, they are some warnings when starting training.
	- Not compatible with TPU, since google's TPU donot implement `tf.data.Dataset.from_generator`
	- Currently, the model work for chinese language only

