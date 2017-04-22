import scipy, numpy
#from sklearn import neighbors, datasets
from sklearn.svm import LinearSVC
import pickle
import csv
import warnings

bball = {}

x = ['superstar','star','starter','role_player','bust']
x = numpy.array(x)
bball['target_names'] = x


with open('college-nba-players-stats.csv') as csvfile:
    nbalist = csv.reader(csvfile)
    data = []
    target = []
    for row in nbalist:
        z = []
        for y in range(1,13):
            z.append(float(row[y]))
        data.append(z)
        target.append(float(row[13]))

data = numpy.array(data)
bball['data'] = data

target = numpy.array(target)
bball['target'] = target

bball['DESCR'] = 'blah'

feature_names = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'personal fouls', 'FT%', '3P%', 'FG%', 'height(cm)', 'SOS']
feature_names = numpy.array(feature_names)
bball['feature_names'] = feature_names

x,y = bball['data'], bball['target']

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    linSVC = LinearSVC()
    linSVC.fit(x,y)
    result = linSVC._predict_proba_lr([13.8,5.4,0.8,0.9,1.6,2.0,4.1,.685,.350,.530,8.12,206])
    print "Marquese Chriss - ", result
    result = linSVC._predict_proba_lr([17.3,3.0,7.0,1.5,0.1,2.0,1.8,.856,.344,.434,8.84,175])
    print "Tyler Ulis - ", result
    result = linSVC._predict_proba_lr([14.6,5.4,2.0,0.8,0.6,3.1,3.2,.654,.294,.431,7.87,201])
    print "Jaylen Brown - ", result

#s = pickle.dumps(linSVC)
#print s
#linSVC2 = pickle.loads(s)
#result = linSVC2._predict_proba_lr([28.6,4.4,5.6,2.5,0.2,3.7,2.38,.876,.387,.454,-3.33,190])


#print result
