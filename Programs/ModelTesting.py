def test_naive_bayes():
    from NaiveBayes import naive_bayes_model
    from LoadData import load_from_file
    
    
    ## Init variables
    number_of_examples = 20000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    
    list = []
    iMax = 50
    for i in range(1,iMax):
        print("running for i = " + str(i))
        x = naive_bayes_model(df, verbose, i)
        
        
        list.append((x[1][0], x[1][1]))
   
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    
    general_acc = [b for  (a,b) in list]
    
    generalisation = [100*(a-b) for (a,b) in list]
    
    plt.scatter(range(1,iMax),generalisation)
   
    plt.show()

    
test_naive_bayes()