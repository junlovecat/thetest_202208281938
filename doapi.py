import nltk,numpy,tflearn,tensorflow,json,pickle
from nltk.stem.lancaster import LancasterStemmer
model=0
bag_of_words=0
words=0
labels=0
data=0
def thisnamenoexistlolwasansyouknowWYSIwasanswayouknow():
    global model,bag_of_words,words,labels,data
    stemmer=LancasterStemmer()
    try:nltk.download('punkt')
    except:pass
    with open("intents.json",encoding='utf-8') as file:data=json.load(file)
    try:
        with open("data.pickle","rb") as f:
            words,labels,training,output=pickle.load(f)
    except:
        words=[]
        labels=[]
        docs_x=[]
        docs_y=[]
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds=nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
        words=[stemmer.stem(w.lower()) for w in words if w != "?"]
        words=sorted(list(set(words)))
        labels=sorted(labels)
        training=[]
        output=[]
        out_empty=[0 for _ in range(len(labels))]
        for x,doc in enumerate(docs_x):
            bag=[]
            wrds=[stemmer.stem(w.lower()) for w in doc]
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row=out_empty[:]
            output_row[labels.index(docs_y[x])]=1
            training.append(bag)
            output.append(output_row)
        training=numpy.array(training)
        output=numpy.array(output)
        with open("data.pickle","wb") as f:
            pickle.dump((words,labels,training,output),f)
    tensorflow.compat.v1.reset_default_graph()
    tensorflow.compat.v1.disable_eager_execution()
    net=tflearn.input_data(shape=[None,len(training[0])])
    net=tflearn.fully_connected(net,8)
    net=tflearn.fully_connected(net,8)
    net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
    net=tflearn.regression(net)
    model=tflearn.DNN(net)
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")
    def bag_of_words(s,words):
        bag=[0 for _ in range(len(words))]
        s_words=nltk.word_tokenize(s)
        s_words=[stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i,w in enumerate(words):
                if w==se:
                    bag[i]=1
        return numpy.array(bag)

from flask import Flask
app=Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/do/<emotion>')
def user(emotion):
    print(type(model.predict([bag_of_words(emotion,words)])[0]))
    return f'{model.predict([bag_of_words(emotion,words)])[0]}'

if __name__=='__main__':
    thisnamenoexistlolwasansyouknowWYSIwasanswayouknow()
    app.run(debug=True,host='0.0.0.0',port=1802)