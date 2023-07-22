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

def split_chunks(arr, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + step]

model_name_or_path="/home/g/models/open_llama_7b"
input_dir = "./json_v0.3/"
guestPrefix = "### Human: "
lexPrefix = "### Assistant: "
output_fn="data_v0.3.jsonl"
cutoff_len_tokens = 2000
overlap_len_tokens = 20


print(f"Loading tokenizer ..")
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama'
)
print("done")

jsons_list=import_jsons_from_dir(input_dir)

print("Concatenating conversations..")
multiturns=[]
x=0
for json_single in jsons_list:
	# conversation = 1 episode
	x+=1
	print(f"conversation #{x}")
	conversation=json_single["conversation"]
	turns_collected=[]
	currTokenCnt=0
	currText=""
	for i in range(0, len(conversation)-1):
		# go through turns
		if conversation[i]["text"].strip()=="" or conversation[i]["from"]!= "Guest":
			continue
		else:
			# turn_text = one QA pair
			turn_text=guestPrefix + conversation[i]["text"] + tokenizer.eos_token \
					 + lexPrefix + conversation[i+1]["text"] + tokenizer.eos_token
			turn_tokens = tokenizer.encode(turn_text, add_special_tokens=False)

			# collect QA pairs until cutoff_len_tokens reached
			if(currTokenCnt+len(turn_tokens)>cutoff_len_tokens):
				turns_collected.append(currText)

				currText=turn_text
				currTokenCnt=len(turn_tokens)
			else:
				currTokenCnt+=len(turn_tokens)
				currText+=turn_text
	multiturns.extend(turns_collected)

# Go throuh all the multiturns and chunk the big ones 
multiturns_chunked = []
step = cutoff_len_tokens - overlap_len_tokens
x=0
for multiturn in multiturns:
	x+=1
	print(f"Chunking multiturn #{x}")

	tokens = tokenizer.encode(multiturn, add_special_tokens=False)

	if step <= 0:
		print(f"Error: overlap_len ({overlap_len_tokens}) cannot be greater than or equal to cutoff_len ({cutoff_len_tokens})")

	tokens_split = list(split_chunks(tokens, step))

	for i in range(1, len(tokens_split)):
		tokens_split[i] = tokens_split[i - 1][-overlap_len_tokens:] + tokens_split[i]

	text_chunks = [tokenizer.decode(x) for x in tokens_split]
	for chunk in text_chunks:
		multiturns_chunked.append({
			'text': chunk
		})


x=0
with open(os.path.expanduser(output_fn), "w") as of:
	for txt in multiturns_chunked:
		x+=1
		of.write(json.dumps(txt) + "\n")
print(f"wrote {x} turns to {output_fn}")