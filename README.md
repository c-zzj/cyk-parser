# cyk-parser
A Python implementation of the CYK algorithm based on nltk.

Required package: nltk

Syntax of grammar: https://www.nltk.org/howto/grammar.html

Implementation is based on the pseudocode from Jacob Eisenstein's [book](https://mitpress.mit.edu/books/introduction-natural-language-processing):

![algorithm](https://github.com/c-zzj/cyk-parser/blob/main/algorithm.JPG?raw=true)

The algorithm first converts the grammar to [CNF](https://en.wikipedia.org/wiki/Chomsky_normal_form), then performs the CYK algorithm to construct all possible parse trees, and converts the trees back to the original grammar.


`french-grammar.txt` contains a (not really) subset of french grammar for illustration purposes.

Table for abbreviations:

| abbr. | meaning       |          abbr.          | meaning          |
|:-----:|---------------|:-----------------------:|------------------|
|   S   | sentence      |           NP            | noun phrase      |
|  PR   | pronoun       |           PN            | proper noun      |
|  VP   | verb phrase   |            V            | verb             |
|   N   | noun          |           DT            | determiner       |
|   A   | adjective     |           DO            | direct object    |
|  ADV  | adverb        |           NE            | negation         |
|  PRE  | placed before |           AFT           | placed after     |
| MASC  | masculine     |           FEM           | feminine         |
| SG | singular |           PL            | plural           |
| SUBJ | subject |           OBJ           | object           |
| RA | requires an article |           WA            | without an article |
| 1 | first person |            2            | second person    |
| 3 | third person |

