import nltk
import sys

nltk.download('punkt')


# Terminal and nonterminal definitions (already provided)
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
NP -> N | Det N | Det Adj N | Det Adj Adj N | N P Det N | N P Det Adj N
VP -> V | V NP | V NP PP | V PP | V NP PP PP
PP -> P NP
"""

# Creating a CFG from the grammar rules
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.leaves()))


def preprocess(sentence):
    words = nltk.word_tokenize(sentence.lower())
    words = [word for word in words if any(c.isalpha() for c in word)]
    return words


def np_chunk(tree):
    noun_phrases = []

    # Traverse the tree and collect NP subtrees
    for subtree in tree.subtrees(lambda t: t.label() == 'NP'):
        # Check if this subtree has no nested NP subtrees
        if not list(subtree.subtrees(lambda t: t.label() == 'NP' and t != subtree)):
            noun_phrases.append(subtree)

    return noun_phrases



if __name__ == "__main__":
    main()
