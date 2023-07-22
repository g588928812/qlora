import json
import os
import glob
from transformers import AutoTokenizer

def import_json(file_path):
    file_path = os.path.expanduser(file_path)
    f = open(file_path)
    data = json.load(f)
    f.close()

    return data

def import_jsons_from_dir(dir_path):
    files = glob.glob(os.path.expanduser(dir_path) + "/*.json")

    jsons_list=[]
    for f in files:
        jsons_list.append(import_json(f))

    return jsons_list

model_name_or_path="models/open_llama_7b"
input_dir = "./json/"
guestPrefix = "### Human: "
lexPrefix = "### Assistant: "
output_fn="data_v0.3.jsonl"
cutoff_len_tokens = 2000
overlap_len_tokens = 20

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama'
)

jsons_list=import_jsons_from_dir(input_dir)

multiturns=[]
x=0
for json_single in jsons_list:
	# conversation = 1 episode
	x+=1
	filename=json_single["filename"]
	conversation=json_single["conversation"]
	# print(f"\nconversation #{x}: {filename}")
	lastO=""
	sum1=0
	sum2=0
	for i in range(0, len(conversation)-1):
		# go through turns
		if conversation[i]["from"]== "Lex":
			lastO=conversation[i]["text"]
			sum1+=len(conversation[i]["text"])
		else:
			sum2+=len(conversation[i]["text"])
	# print(lastO)
	if sum2>0:
		ratio=sum1/sum2
	else:
		ratio=10000

	if(ratio>1):
		print(f"{ratio} {filename}")



