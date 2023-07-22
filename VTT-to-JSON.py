import webvtt
import json
import sys
from os.path import exists

if len(sys.argv) != 2:
    print("No filename given")
    exit()

filename=sys.argv[1]

if not exists(filename):
    print(f"File {filename} does not exist")
    exit()

print(f"processing {filename}")

episode={}
episode["filename"]=filename
conversation=[]

speakermap={}
textBuff=None
currSpeaker=None
for caption in webvtt.read(filename):
    speaker=caption.text.split(" ")[0]
    text=" ".join(caption.text.split(" ")[1:])

    if speaker not in speakermap:
        speakermap[speaker]="Guest"

    if currSpeaker is None:
        textBuff = text
        speakermap[speaker]="Lex"
    elif speaker==currSpeaker:
        textBuff = textBuff + " " + text
    else:   
        # print(f"{currSpeaker} {textBuff}")
        conversation.append({
            "from": speakermap[currSpeaker],
            "text": textBuff
            })
        textBuff = text
    currSpeaker = speaker

episode["conversation"]=conversation

print(f"number of speakers: {len(speakermap)}")
for speaker in speakermap:
    print(f"{speakermap[speaker]} (={speaker})")

outfn=filename + ".json"
with open(outfn, 'w') as fp:
    json.dump(episode, fp)

print(f"output written to {outfn}")
