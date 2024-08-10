from gensim.models import KeyedVectors
import pickle


onto_path = 'elementkgontology.embeddings.txt'
ontoemb = KeyedVectors.load_word2vec_format(onto_path, binary=False)


with open('../chemprop/data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]

# get functional group -> embedding dict
print("getting fg2emb dict ...")
fg2emb = {}
for fg in name:
    fg_name = "http://www.semanticweb.org/ElementKG#"+ fg
    ele_emb = ontoemb[fg_name]
    fg2emb[fg] = ele_emb
pickle.dump(fg2emb, open('fg2emb.pkl','wb'))
print("fg2emb dump finish!")
