import csv
import cPickle as pickle


def most_common_feature(f_name, data):
	# Count how many dws there are
	feature_count = dict()
	for k,v in data.iteritems():
		feature = v[f_name]
		if feature in feature_count:
			feature_count[v] += 1
		else:
			feature_count[v] = 1


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


if __name__ == '__main__':
	data = read_csv_file()
	most_common_feature('dws', data)