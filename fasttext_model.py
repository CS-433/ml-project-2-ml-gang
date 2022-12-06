from fasttext import train_supervised 

def fasttext_label(df):
    df = df["label"].apply(lambda label: "__label__0" if label == -1 else "__label__1")
    return df    

def train_test_fasttext_model(df_train,df_test):
    train_file = 'data/fasstext_train.txt'
    df_train.to_csv(train_file, header=None, index=False, sep=' ', columns=["label","tweets"])
    test_file =  'data/fasstext_test.txt'
    df_test.to_csv(test_file, header=None, index=False, sep=' ', columns=["tweets"])    

    model = train_supervised(input=train_file)

    predictions = []
    file1 = open('data/fasstext_test.txt', 'r')
    lines = file1.readlines()
    for line in lines:
        pred = model.predict(line.strip())[0][0]
        if pred == "__label__0":
            predictions.append(-1)
        else:    
            predictions.append(1)
    return predictions        