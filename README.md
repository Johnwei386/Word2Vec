Word2vec model supported Chinese
====

This is a implement for word2vec to generate the embeddings for Chinese. It is based on the work of CS224n, a class for  Natural Language Processing with Deep Learning. 

Requirement:
======
	Python: 2.7.15   
	tensorflow-gpu: 1.8.0 

Input format:
======
CoNLL format (prefer BMES tag scheme, compatible with the Chinese word segmentation scheme), with each character its label for one line. Sentences are splited with a null line.

	中	B
	国	E
	财	B
	团	E
	买	B
	下	E
	AC	B
	米	M
	兰	E

	代	B
	价	E
	超	S
	10	B
	亿	E
	欧	B
	元	E

How to run the code?
====
1. python main.py -d (datasets path) -t (where to save generate the wordvectors)

