import csv
import cPickle as pickle
import numpy as np

import operator

import matplotlib.pyplot as plt

import plotly

plotly.offline.init_notebook_mode()

import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

import plotly.offline as offline


def get_top_features_by_count(data, count):
	print('Get top features by count ' + str(count))
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


def get_pairs_for_feature(features, data, f1, f2):
	pair_counts = dict()

	for f in features:
		pair_counts[f] = dict() 	# initialize empty dictionary for each top feature

	f_set = set(features)
	for k,v in data.iteritems():
		feature_set = v[f1] 					# Set of features for type
		set_intersection = feature_set & f_set	# Set intersection to see if desired features are contained
		for f in set_intersection:
			for g in v[f2]:
				if g in pair_counts[f]:
					pair_counts[f][g] += 1
				else:
					pair_counts[f][g] = 1

	pickle.dump(pair_counts, open('pair_count_' + f1 + '_' +f2 + '.p', 'wb'))

	return pair_counts


def get_max_pairs(f_name):
	data = pickle.load(open(f_name, 'rb'))
	pairs = dict()
	max_match = -1
	labels = (dict(), dict())
	id_1 = 0
	id_2 = 0
	for k,v in data.iteritems():

		if k not in labels[0]:
			labels[0][k] = id_1
			id_1 += 1

		sorted_pairs = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
		pair, count = zip(*sorted_pairs)
		max_count = count[0]
		arr = np.array(count)
		indices = np.where(arr == max_count)[0] # Get all matches with the maximum count

		pair_features = [pair[i] for i in indices] # Matches with the maximum count

		for feature in pair_features:	# Give those hashes labels
			if feature not in labels[1]:
				labels[1][feature] = id_2
				id_2 += 1

		pairs[k] = (pair_features, max_count) # (Hash matches, maximum count)

		if max_count > max_match:
			max_match = max_count

	pickle.dump((pairs, max_match), open('max_' + f_name, 'wb'))
	pickle.dump(labels, open('pair_ids_' + f_name, 'wb')) # Number for each hash that is a high frequency hash as well as a matching hash


def create_feature_vectors_for_pairs(data, labels, pickle_name):

	feature_matrix = np.zeros(shape=(len(labels[0]), len(labels[1])))
	pairs = data[0]
	max_match = data[1]
	for feature, matches in pairs.iteritems(): # Feature is the hash, matches is (hash matches, num_matches)
		feature_vec = np.zeros(len(labels[1]))
		num_matches = matches[1] # Number of matches
		for match in matches[0]:
			feature_vec[labels[1][match]] = num_matches 	# Enter num matches into index of matching feature
		feature_matrix[labels[0][feature], :] = feature_vec

	pickle.dump(feature_matrix, open(pickle_name, 'wb'))


def create_co_var_matrix(data, file_name):
	print('Creating co variance matrix')
	co_matrix = np.zeros(shape=(len(data), len(data)))
	closest = float('inf')
	farthest = -1
	for i in range(0, len(data)):
		vec_i = normalize_vec(data[i])
		for j in range(0, len(data)):
			vec_j = normalize_vec(data[j])
			dist = np.dot(vec_i, vec_j)
			if dist < closest:
				closest = dist
			if dist > farthest:
				farthest = dist
			co_matrix[i][j] = dist

	pickle.dump((co_matrix, closest, farthest), open(file_name, 'wb'))


def normalize_vec(vector):
	vector = np.asarray(vector)
	norm = np.linalg.norm(vector) # Normalize vectors
	return vector/norm


def heat_map(data, title, min=0, max=1):

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.set_title(title)

	plotly_fig = tls.mpl_to_plotly(fig)

	trace = dict(
		z=data,
		type='heatmap',
		zmin=min,
		zmax=max
	)

	plotly_fig['data'] = [trace]

	plotly_fig['layout']['xaxis'].update({'autorange': True})
	plotly_fig['layout']['yaxis'].update({'autorange': True})
	plotly_fig['layout']['width'] = 1300
	plotly_fig['layout']['height'] = 1000

	plot_url = offline.plot(plotly_fig, filename='heat_map_' + title)


def build_unique_feature_set(data):

	unique_dws = {}
	unique_dns = {}
	unique_so = {}

	label_dws = 0
	label_dns = 0
	label_so = 0

	# Encode labels for features

	for uuid, properties in data.iteritems():
		dws_features = properties['dws']
		dns_features = properties['dns']
		so_features = properties['so']

		for dws in dws_features:
			if dws not in unique_dws:
				unique_dws[dws] = label_dws
				label_dws += 1
		for dns in dns_features:
			if dns not in unique_dns:
				unique_dns[dns] = label_dns
				label_dns += 1
		for so in so_features:
			if so not in unique_so:
				unique_so[so] = label_so
				label_so += 1

	pickle.dump(unique_dws, open('uniq_dws.p', 'wb'))
	pickle.dump(unique_dns, open('uniq_dns.p', 'wb'))
	pickle.dump(unique_so, open('uniq_so.p', 'wb'))


def labels_to_textfile(fname, lbls):
	f = open(fname, 'w')
	for k,v in lbls[0].iteritems():
		f.write(k + ' : ' + str(v) + '\r\n')
	f.close()



if __name__ == '__main__':

	# ========== Load data and counts ==========  #
	data = read_csv_file()
	# build_unique_feature_set(data)

	# # Count most common feature.
	#count_feature('dws', data)
	#count_feature('dns', data)
	#count_feature('so', data)
	#count_feature('license_id', data)

	#dws_counts = pickle.load(open('count_dws.p', 'rb'))
	#sort_by_count('dws', dws_counts)
	sorted_dws = pickle.load(open('sorted_dws.p', 'rb'))
	top_features = get_top_features_by_count(sorted_dws, 20)

	# =========== Get top feature for DWS =========== #

	#get_pairs_for_feature(top_features, data, 'dws', 'dws')
	# get_max_pairs('pair_count_dws_dws.p')
	# max_pairs = pickle.load(open('max_pair_count_dws_dws.p', 'rb'))
	# labels = pickle.load(open('pair_ids_pair_count_dws_dws.p', 'rb'))
	# create_feature_vectors_for_pairs(max_pairs, labels, 'max_pairs_f_vec_dws_dws.p')
	# print('Load Maximum pair feature vectors for dws dws')
	# f_vec_pairs = pickle.load(open('max_pairs_f_vec_dws_dws.p', 'rb'))
	# create_co_var_matrix(f_vec_pairs, 'covar_pairs_dws_dws.p')
	# co_var_matrix, min, max = pickle.load(open('covar_pairs_dws_dws.p', 'rb'))
	# heat_map(co_var_matrix, 'pair similarity dws dws', min, max)
	# labels_to_textfile('dws_dws_labels.txt', labels)

	# ============ Get top feature pair for DWS and DNS ========= #
	# get_pairs_for_feature(top_features, data, 'dws', 'dns')
	# get_max_pairs('pair_count_dws_dns.p')
	# max_dws_dns_pairs = pickle.load(open('max_pair_count_dws_dns.p', 'rb'))
	# dws_dns_labels = pickle.load(open('pair_ids_pair_count_dws_dns.p', 'rb'))
	# create_feature_vectors_for_pairs(max_dws_dns_pairs, dws_dns_labels, 'max_pairs_f_vec_dws_dns.p')
	# f_vec_pairs = pickle.load(open('max_pairs_f_vec_dws_dns.p', 'rb'))
	# create_co_var_matrix(f_vec_pairs, 'covar_pairs_dws_dns.p')
	# co_var_matrix_dws_dns, min, max = pickle.load(open('covar_pairs_dws_dns.p', 'rb'))
	# heat_map(co_var_matrix_dws_dns, 'pair similarity dws dns', min, max)
	# labels_to_textfile('dws_dns_labels.txt', dws_dns_labels)
	# heat_map('max_pair_count_dws_dns.p', 'dws with dns')

	# ============ Get top feature pair for DWS and SO ========= #
	get_pairs_for_feature(top_features, data, 'dws', 'so')
	get_max_pairs('pair_count_dws_so.p')
	max_dws_so_pairs = pickle.load(open('max_pair_count_dws_so.p', 'rb'))
	dws_so_labels = pickle.load(open('pair_ids_pair_count_dws_so.p', 'rb'))
	create_feature_vectors_for_pairs(max_dws_so_pairs, dws_so_labels, 'max_pairs_f_vec_dws_so.p')
	f_vec_pairs = pickle.load(open('max_pairs_f_vec_dws_so.p', 'rb'))
	create_co_var_matrix(f_vec_pairs, 'covar_pairs_dws_so.p')
	co_var_matrix_dws_so, min, max = pickle.load(open('covar_pairs_dws_so.p', 'rb'))
	heat_map(co_var_matrix_dws_so, 'pair similarity dws so', min, max)
	labels_to_textfile('dws_so_labels.txt', dws_so_labels)
	heat_map('max_pair_count_dws_so.p', 'dws with so')