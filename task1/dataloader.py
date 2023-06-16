import csv
import sys
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from ML.rdchiral import rdchiral
from rdchiral.template_extractor import extract_from_reaction

class RPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        return X, y

def load_from_csv(path):
    data = []
    template_dict = {}

    with open('template_id.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            template, template_id = row
            template_dict[template] = int(template_id)

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            _id, class_name, reaction = line
            reactants, products = reaction.split('>>')
            
            inputRec = {'_id': _id, 'reactants': reactants, 'products': products}
            ans = extract_from_reaction(inputRec)
            template = ans['reaction_smarts']

            mol = Chem.MolFromSmiles(products)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits())
            arr[onbits] = 1

            if template in template_dict:
                template_id = template_dict[template]
            else :
                KeyError("No template id!")
            data.append((arr, template_id))

    return data

if __name__== '__main__':
    path = 'testloader.csv'
    data = load_from_csv(path)
    dataset = RPDataset(data)
