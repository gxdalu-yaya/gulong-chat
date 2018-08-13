#coding=utf-8
import sys

first_flag = True
last_utterance = ""
for line in sys.stdin:
    line = line.strip()
    if first_flag == True:
        first_flag = False
        last_utterance = line
        continue
    elif line == "":
        first_flag = True
        last_utterance = line
        continue
    elif last_utterance != "":
        print(last_utterance+"\t"+line)
    last_utterance = line
