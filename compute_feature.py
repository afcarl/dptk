import json
import random


def createPairs(data1,data2, filename,filename1):
    with open(data1) as json_data:
        d = json.load(json_data)
    with open(data2) as json_data:
        d1 = json.load(json_data)
    print len(d)
    print len(d1)
    count = 0
    count1 = 50
    count2 = 50
    nonpairs = []
    dataset1 = set()
    for k,v in d.items():
        dataset1.add(k)

    with open(filename1, mode='w') as output1:
        with open(filename, mode='w') as output:
            for k,v in d.items():
                if k in "num_record":
                    continue
                for k1,v1 in d1.items():
                    if k1 in "num_record":
                        continue
                    if k == k1:
                        output.write(k + "\t" + k1 + "\tSame pair" + "\n")
                    else:
                        if k1 in dataset1:
                            nonpairs.append(k + "\t" + k1 + "\tNot Same pair")
                        #output1.write(k + "\t" + k1 + "\tNot Same pair" + "\n")
        random.shuffle(nonpairs)
        for i in range(0,len(d)):
            output1.write(nonpairs[i] + "\n")


def createCrossPairs(data1,data2, filename):
    with open(data1) as json_data:
        d = json.load(json_data)
    with open(data2) as json_data:
        d1 = json.load(json_data)
    print len(d)
    print len(d1)
    count = 0
    count1 = 50
    count2 = 50
    nonpairs = []

    with open(filename, mode='w') as output:
        for k,v in d.items():
            if k in "num_record":
                continue
            for k1,v1 in d1.items():
                if k1 in "num_record":
                    continue
                nonpairs.append(k + "\t" + k1 + "\tUnknown")
                    #output1.write(k + "\t" + k1 + "\tNot Same pair" + "\n")
        random.shuffle(nonpairs)
        for i in range(0,len(nonpairs)):
            output.write(nonpairs[i] + "\n")

if __name__ == '__main__':
    #createPairs('data_stats1.json','data_stats2.json','pairs.txt','non_pairs.txt')
    createPairs('university_profile.json','university_profile1.json','university_pairs.txt','university_non_pairs.txt')
    createPairs('Organisation_profile5k.json', 'Organisation_profile5k1.json','pairs1.txt','non_pairs1.txt')
