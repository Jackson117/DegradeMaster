import sys
import os

from openbabel import pybel
from rdkit.Chem import AllChem, Draw
from rdkit import Chem

import pandas as pd
import requests
import re
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import mol2graph, pdb_to_mol2

import pymol

def smile2mol(smile, name, comp_id, path):

    name = name + '_' + comp_id

    mol = pybel.readstring('smi', smile)
    mol.addh()
    mol.make3D()
    mol.write("mol2", "./data/case_study/{}/{}_lig.mol2".format(path, name), overwrite='True')

    m=Chem.MolFromMol2File("./data/case_study/{}/{}_lig.mol2".format(path, name),sanitize=False)
    Draw.MolToImage(m)

def getHTML(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        print("Download Succeed!")
        return response.text
    except:
        return "Error Raise!"

def download_PDB(idx, pdb_id, uniport_tar, name_mol):
    # https://files.rcsb.org/download/5MQ1.pdb
    url = "https://files.rcsb.org/download/{}.pdb".format(str(pdb_id))
    print(url)

    if str(pdb_id) == 'None':
        raise Exception("PDB ID cannot be None")

    pdb_text = getHTML(url)
    # print(pdb_text)
    with open('./Warhead_Docking/target_pdb/'+ '{}_{}_{}'.format(idx, pdb_id, uniport_tar) + ".pdb", "w+") as f:
        f.write(pdb_text)
        f.close()

def download_alpha_fold(path, choice):

    f1 = pd.read_csv(path, header=0, encoding='utf-8')
    item_list = f1[['Uniprot', 'Target']].values.astype(str)

    for i, item in enumerate(item_list):
        print('\n{}/{}: '.format(i, len(item_list)))
        uniprot, target = item
        match = re.search(r'\b(\w+)\s', target)

        if match:
            print("First target:", match.group(1))
            target = match.group(1)

        if uniprot == 'nan' or target == 'nan':
            print('uniprot or target missing!')
            continue
        url = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb".format(uniprot)
        print(url)
        pdb_text = getHTML(url)


        with open('./data/PROTAC/{}/{}_{}.pdb'.format(choice, target, uniprot), "w+") as f:
            f.write(pdb_text)
            f.close()

def compare_smiles(smiles1, smiles2):
    # Convert SMILES to molecules and get canonical SMILES
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    canonical_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
    canonical_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

    # Check if they are identical
    if canonical_smiles1 == canonical_smiles2:
        return True
    else:
        return False

def numeric_label(lbl):
    if lbl == 'Active':
        return 1
    elif lbl == 'Inactive':
        return 0
    else:
        raise Exception('Label must be either "Active" or "Inactive with input {}."'.format(lbl))

def analyze_dock_res(path):
    f1 = pd.read_csv(path, header=0, encoding='utf-8')
    inf_list = f1[['output file name', 'Affinity Score']].values.astype(str)

    score_dict = {}
    for item in inf_list:
        name, score = item[0], item[1].astype(float)
        if score_dict.get(name) is None:
            score_dict[name] = [score]
        else:
            score_dict[name].append(score)

    mean_l, var_l = [], []
    for key, value in score_dict.items():
        mean_l.append(np.mean(score_dict[key]))
        var_l.append(np.var(score_dict[key]))
    mean_l, var_l = np.array(mean_l), np.array(var_l)

    cnt_p, cnt_n = 0, 0
    for m in mean_l:
        if m > 0:
            cnt_p += 1
        else:
            cnt_n += 1
    print('postive score: {}'.format(cnt_p))
    print('negative score: {}'.format(cnt_n))

    # index_n = np.where(mean_l < 0, True, False)
    # mean_l, var_l = mean_l[index_n], var_l[index_n]
    # np.random.seed(15)
    # mean_l = np.random.choice(mean_l, 100, replace=False)
    # var_l = np.random.choice(var_l, 100, replace=False)

    plt.hist(mean_l, bins=1000)
    plt.ylim(0, 110)
    plt.xlim(-15, 30)
    plt.ylabel('Frequency')
    plt.xlabel('Mean Score')
    plt.title('Frequency of mean POI ligand docking score')
    plt.show()

    # plt.errorbar(np.arange(len(mean_l)), mean_l, yerr=var_l, marker='o', linestyle='None', capsize=5)
    # plt.ylim(-15,20)
    # plt.ylabel('Docking Score')
    # plt.title('Mean and Variance of POI ligand Docking score')
    # plt.show()

def analyze_mol2graph(path):
    LIGAND_ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
    graph = mol2graph(path, LIGAND_ATOM_TYPE)
    print(graph)
    print('x: ', graph.x)
    print('edge_index: ', graph.edge_index)

def pdb2mol2(pdb_path):
    pymol.cmd.load(pdb_path)
    mol2_path = os.path.basename(os.path.splitext(pdb_path)[0]) + '.mol2'
    pymol.cmd.save(mol2_path)



if __name__ == '__main__':
    # csv_path = './data/PROTAC/warhead.csv'
    # path = 'target_ligand'
    # f1 = pd.read_csv(csv_path, header=0, encoding='utf-8')
    # protac_list = f1[['Compound ID', 'Uniprot', 'Target', 'Smiles']].values.astype(str)
    # for item in protac_list:
    #     comp_id, uniprot, target, smile = item
    #     if comp_id == 'nan' or uniprot == 'nan' or target == 'nan':
    #         continue
    #     name = target + '_' + uniprot
    #     try:
    #         smile2mol(smile, name, comp_id, path)
    #     except:
    #         print(name, comp_id)

    # download_alpha_fold('./data/PROTAC/e3_ligand.csv', 'ligase_pocket')

    # f1 = pd.read_csv('./data/PROTAC/protacdb_20220210.csv', header=0, encoding='utf-8')
    # smiles_list = f1[['PROTAC SMILES', 'Active/Inactive']].values.astype(str)
    # with open('./data/PROTAC/name.json', 'r') as f:
    #     name_dic = json.load(f)
    #
    # added_label = 0
    # for key, value in tqdm(name_dic.items()):
    #     smiles1 = value['link_smile']
    #     for item in smiles_list:
    #         smiles2, label = item
    #         label = numeric_label(label)
    #         if compare_smiles(smiles1, smiles2):
    #             print('Smile matched! | pedia Label: {} | DB Label: {}'.format(label, value['label']))
    #             if value['label'] == -1:
    #                 added_label += 1
    #             break
    # print('Added label: {}'.format(added_label))

    # analyze_dock_res('/Users/a1234809/Documents/Code/Process_Dock_Res/POI/All_Affinity_Score.csv')
    # analyze_mol2graph('./data/PROTAC/ligase_ligand/CRBN_Q96SW2_2_lig_smina_out.mol2')
    # pdb2mol2('./data/PROTAC/target_pocket/BRD2_P25440.pdb')

    smiles = {'dtag-13': 'CC[C@H](C(=O)N1CCCC[C@H]1C(=O)O[C@H](CCC1=CC=C(OC)C(OC)=C1)C1=CC=CC=C1OCC(=O)NCCCCCCOC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O)C1=CC(OC)=C(OC)C(OC)=C1',
              'NanoTag': 'CCC1=CC(=CC=C1)N(CC)C(=O)CN2C3=CC=CC=C3C=C2C(=O)NC4CCC(CC4)C(=O)NCCCCCCCC(=O)NCCCCNC5=CC=CC6=C5C(=O)N(C6=O)C7CCC(=O)NC7=O',
              'HaloPROTAC-3': 'CC1=C(SC=N1)C2=CC(=C(C=C2)CNC(=O)[C@@H]3C[C@H](CN3C(=O)[C@H](C(C)C)N4CC5=CC=CC=C5C4=O)O)OCCOCCOCCOCCCCCCCl',
              'Bromo': 'CC1=C(SC=N1)C2=CC=C(CNC([C@@H]3C[C@@H](O)CN3C([C@H](C(C)(C)C)NC(COCCOCCOCCOC([C@H](CC)[C@H]4C5=NN=C(C)N5C6=C(C(C)=C(C)S6)C(C7=CC=C(Cl)C=C7)=N4)=O)=O)=O)=O)C=C2'}

    i = 0
    for key, val in smiles.items():
        smile2mol(val, key, str(i), 'protac')
        i += 1