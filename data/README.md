# Notes about the Sumerian Text
Sumerian text is originally written in cuneiform symbols and transliterated
to latin characters based on its phonetics (before it is ever translated).
Thus, the transliteration process reveals certain oddities about the language
that appear within the text file. These text files following the ATF convention
(ASCII Translation Format). More information can be found here:
[CDLI ATF Conventions](https://cdli.ucla.edu/support-cdli)

## Syllabograms
A syllabogram is simply a set of characters representing an english sound or
syllable, such as lu, ku,  or uruda. These syllabograms comprise a sumerian
word, and are seperated in the text file by hyphens. Words themselves are
separated by spaces. E.g. lu-uruda-ku ku-uruda-lu2 is 2 words and
6 syllabograms total

## Homophones
Sumerian contains numerous homophones: words with the same pronunciation but
different meanings. Thus, to differentiate between different meanings of the
same word, a number is often used specify which meaning a word is referring to.
The numbering starts at 2 so for 4 different meanings of the word "lu", you
would have: lu, lu2, lu3, and lu4.

## Logograms
Logograms are signs that represent an entire word or concept. Similar to many
mandarin symbols, logograms are a single symbol, that is transliterated into
a syllable representing a whole word/concept. In transliteration, logograms are
denoted with the use of UPPERCASE. For example ti is simply a syllabogram
comprising a larger word while TI is the logogram representing "life".

## Determinatives
Determinatives are similar to determiners in english. They are single letters
or syllables that give meaning to a word in front of or behind the
determinative. The most common determinative is "d", which gives the meaning of
"deity" to whatever word it is adjacent to. Determinatives are not pronounced
and only exist to provide additional meaning. In the dataset, determinatives
appear between braces, e.g. {d}
