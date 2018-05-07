import midi
import re
import argparse
import glob
import operator
import numpy
import sklearn
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report as report

def extract_notes(midi_read):
	#Midi read is a string will all midi file information
	
	#Find Channel with most ticks (most significant channel) 	
	channel_re = r'midi\.NoteOnEvent\(tick=\d+, channel=(\d+), data=\[\d+, \d+\]\)'
	found = re.findall(channel_re,midi_read)
	channel_dict = {f: 0 for f in found}
	for f in found:
		channel_dict[f] += 1
	sig_channel = max(channel_dict.items(), key=operator.itemgetter(1))[0]
	
	#Find each NoteOnEvent note pitch
	pitch_re = r'midi\.NoteOnEvent\(tick=\d+, channel=' + sig_channel + r', data=\[(\d+), \d+\]\)'
	found = re.findall(pitch_re,midi_read)
	pitches = [match for match in found]

	#Return list of integer pitches
	notes = [int(p) for p in pitches]
	return notes

def pitches_n_gram(pitches,n):
	pitches_split = []	
	for i in range(0, len(pitches),n):
		pitches_split.append(pitches[i:i+n])
	return pitches_split
	
def avg_max_pitch(pitches, n):
	#Partition the pitches into n-grams
	grouped_pitches = pitches_n_gram(pitches,n)
	total = 0
	for g in grouped_pitches:
		total += max(g)
	return (total / len(grouped_pitches))

def avg_min_pitch(pitches, n):
	#Partition the pitches into n-grams
	total = 0
	grouped_pitches = pitches_n_gram(pitches,n)
	for g in grouped_pitches:
		total += min(g)
	return (total / len(grouped_pitches))

def num_max_pitch(pitches, n):	
	i = 0
	max_pitch = max(pitches)
	grouped_pitches = pitches_n_gram(pitches,n)
	for pitch in grouped_pitches:
		for p in pitch:
			if p == max_pitch:
				i += 1
	return i

def num_min_pitch(pitches, n):
	i = 0
	min_pitch = min(pitches)
	grouped_pitches = pitches_n_gram(pitches,n)
	for pitch in grouped_pitches:
		for p in pitch:
			if p == min_pitch:
				i += 1
	return i

def avg_pitch(pitches):
	return (sum(pitches) / len(pitches))

def max_pitch_diff(pitches):
	return (max(pitches) - min(pitches))

def max_pitch_diff_avg(pitches, n):
	grouped_pitches = pitches_n_gram(pitches,n)
	max_pitches = [max(pitch) for pitch in grouped_pitches]
	min_pitches = [min(pitch) for pitch in grouped_pitches]
	diff_pitches = [(mx - mn) for (mx,mn) in zip(max_pitches,min_pitches)]
	return (sum(diff_pitches) / len(diff_pitches))

def most_common_pitch(pitches):
	return numpy.bincount(pitches).argmax()

def normalized_pitch_diff_avg(pitches):
	common = most_common_pitch(pitches)

	diff_pitches = [(pitch - common) for pitch in pitches]
	return ( sum(diff_pitches) / len(diff_pitches))

def collect_train_data(path_glob):
	chord_training_data = {}
	paths = glob.glob(path_glob)
	for path in paths:
		chord_category = path.split('/')[1]
		title = path.split('/')[2]
		if chord_category not in chord_training_data:
			chord_training_data[chord_category] = []
		print('MIDI File: ' + title + '\n\t\t\tChord Progression: ' + chord_category)
		
		s =repr(midi.read_midifile(path))
		notes = extract_notes(s)	
		
		chord_training_data[chord_category].append(notes)
	return chord_training_data

def collect_test_data(path):
	test_notes = []
	
	print(path)
	files = glob.glob(path)
	for f in files:
		s =repr(midi.read_midifile(f))
		
		notes = extract_notes(s)
		test_notes.append(notes)
	return test_notes

def collect_labels(data_dict):
	labels = []
	for chord_prog in data_dict.keys():
		for entry in data_dict[chord_prog]:
			labels.append(chord_prog)
	return labels

def create_label_dict(data_dict):
	label_dict = {}
	i = 0
	for key in data_dict.keys():
		label_dict[key] = i
		i += 1
	return label_dict

def main(trainpaths, testpath):
	#Load training data in
	train_dict = collect_train_data(trainpaths)	
	print('Training data loaded successfully')
	
	#Load test data in
	test_data = collect_test_data(testpath)
	print('Test data loaded successfully')

	#Collect labels
	labels = collect_labels(train_dict)
	label_dict = create_label_dict(train_dict)
	inv_label_dict = {value: key for key, value in label_dict.items()}

	#Train Vectorizer	
	v = DictVectorizer(sparse=False)
	train_features_list = []
	for chord_prog in train_dict.keys():
		for note_list in train_dict[chord_prog]:
			n = 4
			D = {
			'max_avg_ngrams' : avg_max_pitch(note_list, n),
			'min_avg_ngrams' : avg_min_pitch(note_list, n),
			'num_notes' : len(note_list),
			'num_max_pitch': num_max_pitch(note_list, n),
			'num_min_pitch': num_min_pitch(note_list, n),
			'avg_pitch': avg_pitch(note_list),
			'max_pitch_diff': max_pitch_diff(note_list),
			'max_diff_avg_ngrams': max_pitch_diff_avg(note_list,n),
			'most_freq_pitch': most_common_pitch(note_list),
			'freq_pitch_diff_avg': normalized_pitch_diff_avg(note_list)
			}
			train_features_list.append(D)
	x_train =  v.fit_transform(train_features_list)
	print('Train Features Step Complete')	

	#Vectorize Test data for later use
	test_features_list = []
	for note_list in test_data:
		n = 4
		D = {
		'max_avg_ngrams' : avg_max_pitch(note_list, n),
		'min_avg_ngrams' : avg_min_pitch(note_list, n),
		'num_notes' : len(note_list),
		'num_max_pitch': num_max_pitch(note_list, n),
		'num_min_pitch': num_min_pitch(note_list, n),
		'avg_pitch': avg_pitch(note_list),
		'max_pitch_diff': max_pitch_diff(note_list),
		'max_diff_avg_ngrams': max_pitch_diff_avg(note_list,n),
		'most_freq_pitch': most_common_pitch(note_list),
		'freq_pitch_diff_avg': normalized_pitch_diff_avg(note_list)
		}
		test_features_list.append(D)	
	x_test = v.transform(test_features_list)
	print('Test Features Step Complete')	

	#Train Classifer
	K = knn(n_neighbors = 5)	
	y_train = [label_dict[label] for label in labels]	
	K = K.fit(x_train,y_train)
	print('KNN Classifier Training Step Complete')

	#For report later on
	y_pred = [] 

	#Predict chord progression using KNN
	for x in x_test:
		x_predict = []
		x_predict.append(x)
		predict = K.predict(x_predict)
		print(inv_label_dict[predict[0]])
		y_pred.append(predict[0])
	#Find Precision, Recall, F-1 scores for test data over chord progressions
	target_names = [prog for prog in label_dict.keys()]
	
	#Hardcoded for test data
	y_true = [0,0,1,2,3,4]
	
	#Print Report
	print(report(y_true,y_pred,target_names = target_names))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog = 'midiparse.py')
	parser.add_argument('--train',required = True)
	parser.add_argument('--test',required = False)
	args = parser.parse_args()
	
	main(args.train,args.test)
