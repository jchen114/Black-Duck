import csv
import cPickle as pickle

import operator


def get_top_feature_by_count(data, count):
	f_id, counts = zip(*data)
	f_idx = counts.index(count)
	top_features = f_id[:f_idx-1]
	return top_features


def sort_by_count(f_name, data):
	sorted_data = sorted(data.items(), key=operator.itemgetter(1), reverse=True)
	pickle.dump(sorted_data, open('sorted_' + f_name + '.p', 'wb'))
	return sorted_data


def count_feature(f_name, data):

	try:
		feature_count = pickle.load(open('count_' + f_name + '.p', 'rb'))
	except:
		# Count how many dws there are
		feature_count = dict()
		for k,v in data.iteritems():
			feature = v[f_name]
			for f in feature:
				if f in feature_count:
					feature_count[f] += 1
				else:
					feature_count[f] = 1
		pickle.dump(feature_count, open('count_' + f_name + '.p', 'wb'))
	return feature_count


def read_csv_file():
	try:
		data = pickle.load(open('black_duck.p', 'rb'))
	except:
		data = dict()
		with open('ubc_data_workshop.csv', 'rb') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=';')
			first_row = True
			for row in csv_reader:
				if first_row:
					first_row = False
					continue
				uuid = row[0]
				dws = row[1]
				dns = row[2]
				so = row[3]
				version = row[4]
				license_id = row[5]
				if uuid not in data:
					features = dict()
					features['dws'] = {dws}
					features['dns'] = {dns}
					features['so'] = {so}
					features['version'] = {version}
					features['license_id'] = {license_id}
					data[uuid] = features
				else:
					print('uuid: ' + uuid)
					# See what is inside dws..
					data[uuid]['dws'].add(dws)
					data[uuid]['dns'].add(dns)
					data[uuid]['so'].add(so)
					data[uuid]['version'].add(version)
					data[uuid]['license_id'].add(license_id)
		pickle.dump(data, open('black_duck.p', 'wb'))
	return data


def get_pairs_for_feature(features, data, f_name):
	pair_counts = dict()

	for f in features:
		pair_counts[f] = dict() 	# initialize empty dictionary for each top feature

	f_set = set(features)
	for k,v in data.iteritems():
		feature_set = v[f_name] 				# Set of features for type
		set_intersection = feature_set & f_set	# Set intersection to see if desired features are contained
		for f in set_intersection:
			for g in feature_set:
				if g in pair_counts[f]:
					pair_counts[f][g] += 1
				else:
					pair_counts[f][g] = 1

	pickle.dump(pair_counts, open('pair_count_' + f_name + '.p', 'wb'))

	return pair_counts


def sort_pairs(f_name):
	data = pickle.load(open(f_name, 'rb'))
	pairs = dict()
	for k,v in data.iteritems():
		sorted_pairs = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
		pairs[k] = sorted_pairs
	pickle.dump(pairs, open('sorted_pairs_' + f_name + '.p', 'wb'))



if __name__ == '__main__':

	# ========== Load data and counts ==========  #
	# data = read_csv_file()
	# # Count most common feature.
	# count_feature('dws', data)
	# count_feature('dns', data)
	# count_feature('so', data)
	# count_feature('license_id', data)


	# =========== Get top feature for DWS =========== #
	#dws_counts = pickle.load(open('count_dws.p', 'rb'))
	#sort_by_count('dws', dws_counts)
	#sorted_dws = pickle.load(open('sorted_dws.p', 'rb'))
	#top_features = get_top_feature_by_count(sorted_dws, 10)
	#get_pairs_for_feature(top_features, data, 'dws')
	sort_pairs('pair_count_dws.p')