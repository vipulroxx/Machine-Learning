sample_memo = '''
a b c c b c e e f g b c f g b c h i b c
'''
#
#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the preceding word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text and a word,
#   and returns a dictionary with keys the set of words that come after, whose values are
#   the number of times the key comes after that word.
#   
#   Just use .split() to split the sample_memo text into words separated by spaces.

def NextWordProbability(sampletext,word):
	word_list = sampletext.split()
	dictionary = dict()
	for i in range(len(word_list)-1):	
		if word_list[i]==word:			
			if word_list[i+1] in dictionary:	
				dictionary[word_list[i+1]]+=1	
			elif word_list[i+1] not in dictionary:	
				dictionary[word_list[i+1]]=1	
	return dictionary

print NextWordProbability(sample_memo, "b")












