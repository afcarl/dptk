# dptk
Data profiling toolkit and Schema matching

Requirement:
1) Install langdetect python library.
2) Install numpy python library.
3) Install SPARQLWrapper python library.
4) Install sklearn and matplotlib python machine learning library.

The downloaded_Data and graphs have the example of downloaded dbpedia data, profiled data, generated match/non-match pairs and graphs for evaluation.

# Data Profiling
The data_profiling.py python script profiles the data in the following format:

```json
{
	"field1.field2...": {
		"num_records": "the number of times the field occurs in the dataset",
		"num_blank": "the number of records where the field occurs and has a blank value (after stripping)",
		"language": "language code, en, sp, etc.",
		"length": {
			"character": {
				"average": "average number of chars for non-empty entries",
				"standard-deviation": "standard deviation of the number of chars for non-empty entries"
			},
			"token": {
				"average": "average number of tokens, separating by blank and punctuation",
				"standard-deviation": "token standard deviation"
			}
		},
		"num_integer": "the number of records where the field contains an integer",
		"num_decimal": "the number of records where the field contains a decimal number",
		"num_distinct_values": "the number of distinct values if less than a given parameter K, -1 if number of distinct values is more than K",
		"num_distinct_tokens": "same as num_distinct_values, but computed based on tokens",
		"frequent-entries": {
			"most_common_values": {
				"value-1": "count 1",
				"value-2": "count-2",
				"value-k": "count-4"
			},
			"most_common_tokens": {
				"token-1": "count 1",
				"token-2": "count-2",
				"token-k": "count-4"
			},
			"most_common_punctuation": {
				"token-1": "count 1",
				"token-2": "count-2",
				"token-k": "count-4"
			},
			"most_common_alphanumeric_tokens": {
				"token-1": "count 1",
				"token-2": "count-2",
				"token-k": "count-4"
			},
			"most_common_numeric_tokens": {
				"token-1": "count 1",
				"token-2": "count-2",
				"token-k": "count-4"
			}
		}
	}
}
```
The following steps are required to run data_profiling.py:
1) python data_profiling.py <file_to_be_profiled> <output_file> <top_k>

Example: python data_profiling.py 'university.json' 'university_profile1.json' 20

# Download and Profile Dbpedia data
The dbpedia_sparql downloads the data from the dbpedia, formats it and uses the profiler for data profiling.
The following steps are required to run dbpedia_sparql.py:
1) python dbpedia_sparql.py <class_Name> <offset_length> <limit_length> <download_file_name> <output_file> <top_k>

Example: python dbpedia_sparql.py 'dbo:University' 1000 1000 'university.json' 'university_profile1.json' 20

# Generate match pairs, non matchpairs and cross pairs from dataset.
compute_feature.py file is used to generate txt files of pairs/non-pairs from the two dataset.
Example: createPairs('university_profile.json','university_profile1.json','university_pairs.txt','university_non_pairs.txt')
The four input parameters are:
1) profiled dataset1
2) profiled dataset2
3) output file for same pairs
4) output file for different pairs
Example output:

For same pairs from two dataset (This is required for training data):

$[http://dbpedia.org/ontology/callSign]	$[http://dbpedia.org/ontology/callSign]	Same pair
$[http://dbpedia.org/ontology/deathDate]	$[http://dbpedia.org/ontology/deathDate]	Same pair

For different pairs from two dataset (This is required for training data):

$[http://dbpedia.org/ontology/placeOfBurial]	$[http://dbpedia.org/ontology/occupation]	Not Same pair
$[http://dbpedia.org/ontology/serviceEndYear]	$[http://dbpedia.org/ontology/associatedMusicalArtist]	Not Same pair

The other function is createCrossPairs(ds1, ds2, outputfile) which creates a cross product list of all the fields in two dataset.

$[http://dbpedia.org/ontology/percentageOfAreaWater]	$[http://dbpedia.org/ontology/timeshiftChannel]	Unknown
$[http://dbpedia.org/ontology/foundedBy]	$[http://dbpedia.org/ontology/homeArena]	Unknown

# Random shuffle
random_stats.py is used to shuffle the two database. The data fetched from dbpedia for any two dataset is not random. For example, ds1 for Person would be first 1000 objects and ds2 for Person would be next 1000 objects. Hence, to randomize the data random_stats.py is used before profiling.

# Schema matching
Machine learning approach is used for schema matching. The data is obtained from dbpedia. The features used are from the data profiling sincce data profiler provides statistics for each field in the datasset. Here given two fields from two different dataset, given the profiler for the dataset, a decision is made whether the two fields are same field or not the same field. The classifier (SVM) is trained on the dbpedia data.

There are two separate classifier for numeric data field and non-numeric data field since there are additional features used for numeric data fields. feature.py is the file used to train and test the data. 

This is the api to train the classifier.
clf, clf_numeric = get_classifiers('data_stats1.json', 'data_stats2.json', 'pairs.txt', 'non_pairs.txt')

This api is used to predict the same field or not the same field
predictPairs('Organisation_profile5k.json','Organisation_profile5k1.json',clf,clf_numeric)

This api is used to evaluate the classifier.    predict('university_profile.json','university_profile1.json','university_pairs.txt','university_non_pairs.txt',"University",clf,clf_numeric)
