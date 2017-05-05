from SPARQLWrapper import SPARQLWrapper, JSON
import os, json
import data_profiling
import sys
map = {'ulan':"http://vocab.getty.edu/sparql",
       'aac':"http://data.americanartcollaborative.org/sparql",
       'dbpedia':"http://dbpedia.org/sparql"}

files = os.listdir( os.path.join(os.path.dirname(os.path.realpath(__file__)),'sparql'))


def sparqlQuery(dbo_type,num_samples):
# Iterate over all SPARQL files
    res = {}
    for f in files:
        # Extract museum name

        base = f[:f.index('.')] # ulan, npg etc.
        f_in = open(os.path.join('sparql',f), 'r')

        if base not in "dbpedia":
            continue
        # Send SPARQL query
        sparql = SPARQLWrapper(map['dbpedia'])

        sparql_query = f_in.read()
        sparql_query = sparql_query.replace("<dbo_type>",dbo_type)
        sparql_query = sparql_query.replace("<num_samples>",str(num_samples))

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        f_in.close()

        # Save the results
        #out = open(os.path.join('dataset',base+'.json'),'w')
        res[base]=results["results"]["bindings"]
        '''
        for entity in results["results"]["bindings"]:
            out.write(json.dumps(entity))
            out.write("\n")

        out.close()
        '''
    return res

def sparqlQuery1(instance):
# Iterate over all SPARQL files
    res = {}
    for f in files:
        # Extract museum name

        base = f[:f.index('.')] # ulan, npg etc.

        f_in = open(os.path.join('sparql',f), 'r')
        if base not in "dbpedia_prop_val":
            continue
        # Send SPARQL query
        sparql = SPARQLWrapper(map['dbpedia'])

        sparql_query = f_in.read()
        sparql_query = sparql_query.replace("instance",instance)

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        f_in.close()

        # Save the results
        #out = open(os.path.join('dataset',base+'.json'),'w')
        res[base]=results["results"]["bindings"]
        '''
        for entity in results["results"]["bindings"]:
            out.write(json.dumps(entity))
            out.write("\n")

        out.close()
        '''
    return res

def getListOfObjs(dbo_type, start, num_samples):
    listOfObjects = []
    ans = sparqlQuery(dbo_type,start + num_samples)
    for k,v in ans.iteritems():
        print len(v)
        for res in v:
            if start <= 0:
                listOfObjects.append(res['name']['value'])
            start -= 1
    print len(listOfObjects)
    return listOfObjects

def writeToJson(filename,data):
    cnt = 0
    #with open(filename, mode='w') as feedsjson:
    with open(filename, mode='a') as feedsjson:
        #feedsjson.write('[')
        flag = False
        for inst in listOfObjects:
            print cnt
            cnt += 1
            if flag:
                feedsjson.write(',')
            flag = True
            res = sparqlQuery1(inst)
            #print(res['dbpedia_prop_val'])
            obj = dict()
            for ent in res['dbpedia_prop_val']:
                prop = ent['property']['value']
                if "ontology" in prop:
                    val = ent['value']['value']
                    obj[prop] = val
            json.dump(obj,feedsjson)
        feedsjson.write(']')
def dbpedia_profiled_data(className, offset, limit, dbpediaData, dbpediaProfiledData, top_frequent):
    listOfObjects = getListOfObjs(className,offset, limit)
    writeToJson(dbpediaData,listOfObjects)
    data_profiling.profile_data(dbpediaData,dbpediaProfiledData,top_frequent)

if __name__ == '__main__':
    dbpedia_profiled_data(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
    #listOfObjects = getListOfObjs('dbo:University',3382, 1618)
    #writeToJson('university.json',listOfObjects)
    #data_profiling.profile_data('university1.json','university_profile1.json',20)

