

def cleanText(str):
   from nltk.tokenize import word_tokenize
   chars = [char for char in str if char.isalpha() or char == ' ']
   string = ""
   for c in chars:
        string = string + c
   return string 
   
   
   
#print(cleanText("\\ \Hello, my name is__ oliver!"))

