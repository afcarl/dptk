# dptk
Data profiling toolkit and Schema matching

Requirement:
1) Install langdetect python library.
2) Install numpy python library.
3) Install SPARQLWrapper python library.
4) Install sklearn and matplotlib python machine learning library.

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

The dbpedia_sparql downloads the data from the dbpedia, formats it and uses the profiler for data profiling.
The following steps are required to run dbpedia_sparql.py:
1) python dbpedia_sparql.py <class_Name> <offset_length> <limit_length> <download_file_name> <output_file> <top_k>

Example: python dbpedia_sparql.py 'dbo:University' 1000 1000 'university.json' 'university_profile1.json' 20
