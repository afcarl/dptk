import json
import random
import data_profiling

'''
json1_file = open('data.json').read()
json2_file = open('data1.json').read()

json1_data = json.loads (json1_file)
json2_data = json.loads(json2_file)



for k,v in json1_data.iteritems():
    print k
print json1_data
'''


def random_shuffle(ds1, ds2, dp1, dp2):
    data1_file = open(ds1).read()
    data2_file = open(ds2).read()

    json_data1 = json.loads(data1_file)
    json_data2 = json.loads(data2_file)

    merged_list = json_data1 + json_data2
    random.shuffle(merged_list)
    k = len(merged_list) / 2
    list1 = merged_list[0:k]
    list2 = merged_list[k:]

    with open('data_temp_rand1.json', 'w') as outfile:
        json.dump(list1, outfile)
    with open('data_temp_rand2.json', 'w') as outfile:
        json.dump(list2, outfile)

    data_profiling.profile_data('data_temp_rand1.json', dp1, 20)
    data_profiling.profile_data('data_temp_rand2.json', dp2, 20)


if __name__ == '__main__':
    random_shuffle('university.json', 'university1.json', 'university_profile.json', 'university_profile1.json')
    random_shuffle('Organisation5k.json', 'Organisation5k1.json', 'Organisation_profile5k.json', 'Organisation_profile5k1.json')
