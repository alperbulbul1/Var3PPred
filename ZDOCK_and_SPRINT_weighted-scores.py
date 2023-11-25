import pandas as pd
from statistics import mean
import numpy as np


df1 = pd.read_csv("infever-pathogenic-zdock_screening_results.txt", header=None, sep=",")

df2 = pd.read_csv("infever-nomut-zdock_screening_results.txt", header=None, sep=",")

df3 = df1.groupby(by=[0])[1].apply(list).reset_index(name="listedin")

df4 = pd.read_excel("Infevers-HGPEC-Rank.xlsx", engine='openpyxl')

df5 = pd.read_csv("proteins-enterezid-uniprotid.csv", sep=",")

rank_entezid = df4["EntrezID"].tolist()

rank_score = df4["Rank Score"].tolist()

unprotid = df5["initial_alias"].tolist()

uniprot_to_entezid = df5["name"].tolist()

a = []
b = []

mean_rank = mean(rank_score)

for i, entez1 in enumerate(uniprot_to_entezid):
    for j, entez2 in enumerate(rank_entezid):
        if str(entez1) == str(entez2):
            a.append(unprotid[i])
            b.append(rank_score[j])


extract = set(uniprot_to_entezid)-set(rank_entezid)

for i, enterz in enumerate(extract):
    for j, enterz2 in enumerate(uniprot_to_entezid):
        if str(enterz) == enterz2:
            a.append(unprotid[j])
            b.append(mean_rank)

#print(uniprot_to_entezid)

df6 = pd.DataFrame()

df6["unprotid"] = a
df6["rankscore"] = b

df6 = df6.mask(df6["rankscore"]==0).fillna(df6["rankscore"].min())
df6.drop_duplicates(subset=['unprotid'], inplace=True, ignore_index=True)

df6.to_csv("uniprot-rank-scores.txt", index=False, sep="\t")


mutations = df3[0].tolist()

interaction_mutation = df3["listedin"].tolist()



interaction_scores_list = []
interaction_score_weight_list = []
interaction_scores_list_nomut = []


for i, listed_int in enumerate(interaction_mutation):
    sub_list = []
    sub_list2 = []
    sub_list3 = []
    for prot in listed_int:
        for j, prot2 in enumerate(df6.unprotid):
            if prot2 == prot:

                for k in range(len(listed_int)):
                    filter1 = df1[0].isin([mutations[i]])
                    filter2 = df1[1].isin([listed_int[k]])
                    df9 = df1[(filter1 & filter2)]
                    filter3 = df2[0].isin([mutations[i].split("_")[0]])
                    filter4 = df2[1].isin([listed_int[k]])
                    df10 = df2[(filter3 & filter4)]
                    filter5 = df2[1].isin([mutations[i].split("_")[0]])
                    filter6 = df2[0].isin([listed_int[k]])
                    df11 = df2[(filter5 & filter6)]
                    if not df9.empty:
                        sub_list.append(df1[(filter1 & filter2)].iat[0,2])
                        sub_list2.append(df6.at[j, "rankscore"])
                    if not df10.empty:
                        sub_list3.append(df10.iat[0,2])
                    elif not df11.empty:
                        sub_list3.append(df11.iat[0,2])
    interaction_scores_list.append(sub_list)
    interaction_score_weight_list.append(sub_list2)
    interaction_scores_list_nomut.append(sub_list3)

scores2 = [] 

def DavidRobinson(l):
    return l.count(True) == 1

print(len(interaction_scores_list))
print(interaction_scores_list_nomut)

for i, scores in enumerate(interaction_scores_list):
    if DavidRobinson(scores):
        scores2.append(((scores[0]-interaction_scores_list_nomut[i][0])*interaction_score_weight_list[i][0]))
    else:
        scores = np.array(scores)
        interaction_scores_list_nomut[i] = np.array(interaction_scores_list_nomut[i])
        substracted = np.subtract(scores,interaction_scores_list_nomut[i])
        interaction_score_weight_list[i] = np.array(interaction_score_weight_list[i])
        weighted = np.multiply(substracted, interaction_score_weight_list[i])
        print(weighted)
        scores2.append(np.mean(weighted))

df8 = pd.DataFrame()

df8["Mutations"] = mutations

df8["Weightted_score"] = scores2

print(df8)

df8.to_csv("infever_pathogenic-real-weighted-zdock-scores-final.txt", sep="\t", index=False)
