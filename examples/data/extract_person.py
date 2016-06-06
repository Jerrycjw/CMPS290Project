from stanford_corenlp_pywrapper import CoreNLP
import os
proc = CoreNLP("ner", corenlp_jars=["/Users/Jerry/Downloads/stanford-corenlp-full-2015-12-09/*"])
input_path = '/Users/Jerry/Documents/CMPS290H/Project/data/dataset'
output_path = '/Users/Jerry/Documents/CMPS290H/Project/data/dictionary/name.tsv'
#parse files
output = open(output_path,'w')
for filename in os.listdir(input_path):
	try:
		input_file = open(filename,'r')
		x = input_file.read()
		out = proc.parse_doc(x)
		ner_tags = out['sentences'][0]['ner']
		num_tokens = len(ner_tags)
		lemmas = out['sentences'][0]['lemmas']
		first_indexes = (i for i in xrange(num_tokens) if ner_tags[i] == "PERSON" and (i == 0 or ner_tags[i-1] != "PERSON"))
		for begin_index in first_indexes:
		    # find the end of the PERSON phrase (consecutive tokens tagged as PERSON)
		    end_index = begin_index + 1
		    while end_index < num_tokens and ner_tags[end_index] == "PERSON":
		    	end_index += 1
		    end_index -= 1
		    mention_text = " ".join(map(lambda i: lemmas[i], xrange(begin_index, end_index + 1)))
		    print("%s %s" % (filename, mention_text))
		    output.write("%s\n" % mention_text)
	except IndexError:
		pass