import spacy
"""
Important notes:
1. small language model has no word vectors loaded and similarity is based on tagger, parser and NER
2. unknown words do not have a valid word vector in the larger model and do not produce a meaningful result


"""
nlp = spacy.load('en_core_web_md')
# nlp = spacy.load('en_core_web_sm')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print("")


# tokens = nlp("cat apple monkey banana")

# unknown words do not have a valid word vector and do not produce a meaningful result
tokens = nlp("python java kotlin")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I've lost my car in my car",
             "I'd like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

print(f"\n{sentence_to_compare}")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
