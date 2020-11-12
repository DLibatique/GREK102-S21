import lxml
from bs4 import BeautifulSoup as bs
from itertools import groupby
import pprint
import numpy as np
import matplotlib.pyplot as plt
import re
import requests

url = requests.get("https://raw.githubusercontent.com/PerseusDL/treebank_data/master/v2.1/Greek/texts/tlg0011.tlg004.perseus-grc1.tb.xml")

bs_content = bs(url.text, 'lxml')

# content = []
# with open('soph-ot-treebank.xml', 'r') as file:
#      content = file.readlines()
#      content = "".join(content)
#      bs_content = bs(content, 'lxml')

sentences = bs_content.find_all("sentence")

# all possible combinations of verb aspects
verb_combos = {

    # finite verbs
    "pres_act_ind": r"v(1|2|3)(s|p|d)pia---",
    "pres_act_subj": r"v(1|2|3)(s|p|d)psa---",
    "pres_act_opt": r"v(1|2|3)(s|p|d)poa---",
    "pres_act_imp": r"v(1|2|3)(s|p|d)pma---",
    "pres_mp_ind": r"v(1|2|3)(s|p|d)pie---",
    "pres_mp_subj": r"v(1|2|3)(s|p|d)pse---",
    "pres_mp_opt": r"v(1|2|3)(s|p|d)poe---",
    "pres_mp_imp": r"v(1|2|3)(s|p|d)pme---",
    "impf_act_ind": r"v(1|2|3)(s|p|d)iia---",
    "impf_mp_ind": r"v(1|2|3)(s|p|d)iie---",
    "fut_act_ind": r"v(1|2|3)(s|p|d)fia---",
    "fut_act_opt": r"v(1|2|3)(s|p|d)foa---",
    "fut_mid_ind": r"v(1|2|3)(s|p|d)fim---",
    "fut_mid_opt": r"v(1|2|3)(s|p|d)fom---",
    "fut_pass_ind": r"v(1|2|3)(s|p|d)fip---",
    "fut_pass_opt": r"v(1|2|3)(s|p|d)fop---",
    "aor_act_ind": r"v(1|2|3)(s|p|d)aia---",
    "aor_act_subj": r"v(1|2|3)(s|p|d)asa---",
    "aor_act_opt": r"v(1|2|3)(s|p|d)aoa---",
    "aor_act_imp": r"v(1|2|3)(s|p|d)ama---",
    "aor_mid_ind": r"v(1|2|3)(s|p|d)aim---",
    "aor_mid_subj": r"v(1|2|3)(s|p|d)asm---",
    "aor_mid_opt": r"v(1|2|3)(s|p|d)aom---",
    "aor_mid_imp": r"v(1|2|3)(s|p|d)amm---",
    "aor_pass_ind": r"v(1|2|3)(s|p|d)aip---",
    "aor_pass_subj": r"v(1|2|3)(s|p|d)asp---",
    "aor_pass_opt": r"v(1|2|3)(s|p|d)aop---",
    "aor_pass_imp": r"v(1|2|3)(s|p|d)amp---",
    "perf_act_ind": r"v(1|2|3)(s|p|d)ria---",
    "perf_act_subj": r"v(1|2|3)(s|p|d)rsa---",
    "perf_act_opt": r"v(1|2|3)(s|p|d)roa---",
    "perf_act_imp": r"v(1|2|3)(s|p|d)rma---",
    "perf_mp_ind": r"v(1|2|3)(s|p|d)rie---",
    "perf_mp_subj": r"v(1|2|3)(s|p|d)rse---",
    "perf_mp_opt": r"v(1|2|3)(s|p|d)roe---",
    "perf_mp_imp": r"v(1|2|3)(s|p|d)rme---",
    "plupf_act_ind": r"v(1|2|3)(s|p|d)lia---",
    "plupf_mp_ind": r"v(1|2|3)(s|p|d)lie---",
    "futpf_act_ind": r"v(1|2|3)(s|p|d)tia---",
    "futpf_mp_ind": r"v(1|2|3)(s|p|d)tie---",

    # infinitives
    "pres_act_inf": r"v--pna---",
    "pres_mp_inf": r"v--pne---",
    "fut_act_inf": r"v--fna---",
    "fut_mid_inf": r"v--fnm---",
    "fut_pass_inf": r"v--fnp---",
    "aor_act_inf": r"v--ana---",
    "aor_mid_inf": r"v--anm---",
    "aor_pass_inf": r"v--anp---",
    "perf_act_inf": r"v--rna---",
    "perf_mp_inf": r"v--rne---",
    "futpf_mp_inf": r"v--tne---",

    # participles
    "pres_act_part": r"v-(s|p|d)ppa(m|f|n)(n|g|d|a|v|l)-",
    "pres_mp_part": r"v-(s|p|d)ppe(m|f|n)(n|g|d|a|v|l)-",
    "fut_act_part": r"v-(s|p|d)fpa(m|f|n)(n|g|d|a|v|l)-",
    "fut_mid_part": r"v-(s|p|d)fpm(m|f|n)(n|g|d|a|v|l)-",
    "fut_pass_part": r"v-(s|p|d)fpp(m|f|n)(n|g|d|a|v|l)-",
    "aor_act_part": r"v-(s|p|d)apa(m|f|n)(n|g|d|a|v|l)-",
    "aor_mid_part": r"v-(s|p|d)apm(m|f|n)(n|g|d|a|v|l)-",
    "aor_pass_part": r"v-(s|p|d)app(m|f|n)(n|g|d|a|v|l)-",
    "perf_act_part": r"v-(s|p|d)rpa(m|f|n)(n|g|d|a|v|l)-",
    "perf_mp_part": r"v-(s|p|d)rpe(m|f|n)(n|g|d|a|v|l)-",
    "futpf_mp_part": r"v-(s|p|d)tpe(m|f|n)(n|g|d|a|v|l)-",
}

# get counts for each verb aspect combo
verb_combo_stats = []
for x in verb_combos:
    counter = 0
    for w in bs_content.find_all("word"):
        if w.get("postag"):
            if re.match(verb_combos[x], w.get("postag")):
                counter += 1
    verb_combo_stats.append((x, counter))

# sort most to least
verb_combo_stats = sorted(verb_combo_stats, key=lambda y: y[1], reverse=True)

# # graph verb aspect combo stats
# verb_combo_list = [x[0] for x in verb_combo_stats]
# combo_frequencies = [x[1] for x in verb_combo_stats]
# x_pos = np.arange(len(verb_combo_list))
#
# plt.bar(x_pos, combo_frequencies, align='center')
# plt.xticks(x_pos, verb_combo_list, rotation='vertical')
# plt.ylabel('Number of Occurrences')
# plt.suptitle("Verb Aspect Combos in Sophocles' Oedipus Tyrannus")
# plt.show()


def find_verb_combo(bs_object):
    '''
    function to generate passages with requested verb aspects
    input: beautiful soup scraped xml object
    output: printed passage citations, word form, and postag
    '''
    search_string = r''
    request = input('What part of speech do you want to search?: ')
    if request == 'verb':

        search_string += 'v'

        category = input('finite, infinitive, or participle?: ')

        if category == "infinitive":
            search_string += '--'

            tense = input("What tense? (If it doesn't matter, enter 'x'): ")
            if tense == "present":
                search_string += 'pn'
            elif tense == "future":
                search_string += 'fn'
            elif tense == "aorist":
                search_string += 'an'
            elif tense == "perfect":
                search_string += 'rn'
            elif tense == "future perfect":
                search_string += 'tn'
            elif tense == "x":
                search_string += '(p|f|a|r|t)n'
            else:
                pass

            voice = input("What voice? (If it doesn't matter, enter 'x'): ")
            if voice == "active":
                search_string += 'a---'
            elif voice == "middle":
                search_string += 'm---'
            elif voice == 'passive':
                search_string += 'p---'
            elif voice == 'medio-passive':
                search_string += 'e---'
            elif voice == 'x':
                search_string += '(a|m|p|e)---'
            else:
                pass

        elif category == 'participle':

            search_string += '-'

            number = input("What number? (If it doesn't matter, enter 'x'): ")
            if number == 'singular':
                search_string += 's'
            elif number == 'plural':
                search_string += 'p'
            elif number == 'dual':
                search_string += 'd'
            elif number == 'x':
                search_string += '(s|p|d)'

            tense = input("What tense? (If it doesn't matter, enter 'x'): ")
            if tense == "present":
                search_string += 'pp'
            elif tense == "future":
                search_string += 'fp'
            elif tense == "aorist":
                search_string += 'ap'
            elif tense == "perfect":
                search_string += 'rp'
            elif tense == "future perfect":
                search_string += 'tp'
            elif tense == "x":
                search_string += '(p|f|a|r|t)p'
            else:
                pass

            voice = input("What voice? (If it doesn't matter, enter 'x'): ")
            if voice == "active":
                search_string += 'a'
            elif voice == "middle":
                search_string += 'm'
            elif voice == 'passive':
                search_string += 'p'
            elif voice == 'medio-passive':
                search_string += 'e'
            elif voice == 'x':
                search_string += '(a|m|p|e)'
            else:
                pass

            gender = input("What gender? (If it doesn't matter, enter 'x'): ")
            if gender == 'masculine':
                search_string += 'm'
            elif gender == 'feminine':
                search_string += 'f'
            elif gender == 'neuter':
                search_string += 'n'
            elif gender == 'x':
                search_string += '(m|f|n)'
            else:
                pass

            case = input("What case? (If it doesn't matter, enter 'x'): ")
            if case == 'nominative':
                search_string += 'n-'
            elif case == 'genitive':
                search_string += 'g-'
            elif case == 'dative':
                search_string += 'd-'
            elif case == 'accusative':
                search_string += 'a-'
            elif case == 'vocative':
                search_string += 'v-'
            elif case == 'locative':
                search_string += 'l-'
            elif case == 'x':
                search_string += '(n|g|d|a|v|l)-'
            else:
                pass

        elif category == 'finite':

            person = input("What person? (If it doesn't matter, input 'x'): ")
            if person == 'first':
                search_string += '1'
            elif person == 'second':
                search_string += '2'
            elif person == 'third':
                search_string += '3'
            elif person == 'x':
                search_string += '(1|2|3)'

            number = input("What number? (If it doesn't matter, enter 'x'): ")
            if number == 'singular':
                search_string += 's'
            elif number == 'plural':
                search_string += 'p'
            elif number == 'dual':
                search_string += 'd'
            elif number == 'x':
                search_string += '(s|p|d)'

            tense = input("What tense? (If it doesn't matter, enter 'x'): ")
            if tense == "present":
                search_string += 'p'
            elif tense == "imperfect":
                search_string += 'i'
            elif tense == "perfect":
                search_string += 'r'
            elif tense == "pluperfect":
                search_string += 'l'
            elif tense == "future perfect":
                search_string += 't'
            elif tense == "future":
                search_string += 'f'
            elif tense == "aorist":
                search_string += 'a'
            elif tense == "x":
                search_string += '(p|i|r|l|t|f|a)'
            else:
                pass

            mood = input("What mood? (If it doesn't matter, enter 'x'): ")
            if mood == 'indicative':
                search_string += 'i'
            elif mood == 'subjunctive':
                search_string += 's'
            elif mood == 'optative':
                search_string += 'o'
            elif mood == 'imperative':
                search_string += 'm'
            elif mood == 'x':
                search_string += '(i|s|o|m)'

            voice = input("What voice? (If it doesn't matter, enter 'x'): ")
            if voice == "active":
                search_string += 'a---'
            elif voice == "middle":
                search_string += 'm---'
            elif voice == 'passive':
                search_string += 'p---'
            elif voice == 'medio-passive':
                search_string += 'e---'
            elif voice == 'x':
                search_string += '(a|m|p|e)---'
            else:
                pass

            pass

        else:
            pass
    else:
        pass

    print(search_string)

    counter = 0
    for w in bs_object.find_all("word"):
        if w.get("postag"):
            if re.match(search_string, w.get("postag")):
                counter += 1
                print(w.get("cite") + " | " + w.get("form") + ": " + w.get("postag"))
    print(f"There were {counter} results.")

find_verb_combo(bs_content)

## get sentence.word ID, form, postag

# counter = 0
# for s in sentences:
#     x = {
#         "sent_id": s.get("id"),
#         "sent_lines": s.get("subdoc")
#     }
#     words = s.find_all("word")
#     for w in words:
#         word_id = w.get("id")
#         word_form = w.get("form")
#         word_postag = w.get("postag")
#         word_cite = w.get("cite")
#         if word_id and word_form and word_postag and word_cite and not word_postag.startswith('u'):
#             print(x['sent_id'] + "." + word_id + ": " + word_form + ", " + word_postag)
#             counter += 1
#
# print(counter)

"""
GRAPHS GRAPHS GRAPHS
"""

# get total part of speech counts
pos = 'nvtadlgcrpmieu'
pos_full = ['noun', 'verb', 'participle', 'adjective', 'adverb', 'article', 'particle', 'conjunction', 'preposition', 'pronoun', 'numeral', 'interjection', 'exclamation', 'punctuation']
pos_counts = []
for p in pos:
    pos_counter = 0
    for w in bs_content.find_all("word"):
        if w.get("postag"):
            if p == w.get("postag")[0]:
                pos_counter += 1
    pos_counts.append((p, pos_counter))

## graph parts of speech frequencies
# frequencies = [x[1] for x in pos_counts]
# x_pos = np.arange(len(pos_full))
#
# plt.bar(x_pos, frequencies, align='center')
# plt.xticks(x_pos, pos_full, rotation='vertical')
# plt.ylabel('Number of Occurrences')
# plt.show()

# # graph finite verb person stats
# x_pos = np.arange(3)
# plt.bar(x_pos, [332, 395, 551], align='center')
# plt.xticks(x_pos, ['first', 'second', 'third'])
# plt.ylabel('Number of Occurrences')
# plt.suptitle("Verb Person in Sophocles' OT")
# plt.show()

# # graph finite verb tense stats
# x_pos = np.arange(7)
# plt.bar(x_pos, [569, 356, 139, 126, 83, 3, 2], align='center')
# plt.xticks(x_pos, ['present', 'aorist', 'imperfect', 'future', 'perfect', 'plupf.', 'fut. pf.'])
# plt.ylabel('Number of Occurrences')
# plt.suptitle("Finite Verb Tense in Sophocles' OT")
# plt.show()

# # graph finite verb mood stats
# x_pos = np.arange(4)
# plt.bar(x_pos, [990, 119, 111, 58], align='center')
# plt.xticks(x_pos, ['indicative', 'imperative', 'optative', 'subjunctive'])
# plt.ylabel('Number of Occurrences')
# plt.suptitle("Verb Mood in Sophocles' OT")
# plt.show()

# # graph finite verb voice stats
# x_pos = np.arange(4)
# plt.bar(x_pos, [986, 136, 118, 38], align='center')
# plt.xticks(x_pos, ['active', 'medio-passive', 'middle', 'passive'])
# plt.ylabel('Number of Occurrences')
# plt.suptitle("Verb Voice in Sophocles' OT")
# plt.show()




"""
https://www.kite.com/python/answers/how-to-sort-a-list-of-tuples-by-the-second-value-in-python

Plotting:
- vertical x labels: https://stackoverflow.com/questions/1221108/barchart-with-vertical-labels-in-python-matplotlib
- bar plot from list of tuples: https://stackoverflow.com/questions/13925251/python-bar-plot-from-list-of-tuples
- matplotlib, plot docs: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html
"""
