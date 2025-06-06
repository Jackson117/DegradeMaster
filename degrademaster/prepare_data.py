import os.path
from os.path import basename, splitext, isfile

import torch
import pickle
import pandas as pd
import json
from torch_geometric.data import InMemoryDataset, Data
import torch.nn as nn

from rdkit import Chem
from pathlib import  Path
from Bio.PDB import PDBParser

from degrademaster.dataset import VOCAB, Atom, Block, blocks_to_data
from degrademaster.tools import construct_edges, KNNBatchEdgeConstructor, BlockEmbedding
from degrademaster.config.config import get_args
from openbabel import pybel

from rdkit.Chem import AllChem, Draw
from pymol import cmd
import numpy as np
import subprocess


PROTEIN_ATOM_TYPE =['C','N','O','S']
LIGAND_ATOM_TYPE = ['C','N','O','S','F','Cl','Br','I','P']
ATOMS = [ # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
SMILES_CHAR =['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p']
EDGE_ATTR = {'1':1,'2':2,'3':3,'ar':4,'am':5}
args = get_args()

def trans_smiles(x):
    temp = list(x)
    temp = [SMILES_CHAR.index(i) if i in SMILES_CHAR else len(SMILES_CHAR) for i in temp]
    return temp

def smile2mol(smile, comp_id):

    mol = pybel.readstring('smi', smile)
    mol.addh()
    mol.make3D()
    mol.write("mol2", "./data/PROTAC/protac/{}.mol2".format(comp_id), overwrite='True')

    return f"{comp_id}.mol2"
    #
    # m=Chem.MolFromMol2File("./data/PROTAC/protac/{}_lig.mol2".format(path, name),sanitize=False)
    # Draw.MolToImage(m)

def mol2graph(path, ATOM_TYPE, input_type):
    with open(path) as f:
        lines = f.readlines()
    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n')+1:lines.index('@<TRIPOS>BOND\n')]
    if input_type == 'ligand':
        bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]
    else:
        bond_lines = lines[lines.index('@<TRIPOS>BOND\n')+1:lines.index('@<TRIPOS>SUBSTRUCTURE\n')]
    atoms = []
    for atom in atom_lines:
        if len(atom.split()) < 6:
            print('atom error!')
            print(path)
            continue
        ele = atom.split()[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele) 
                        if ele in ATOM_TYPE 
                        else len(ATOM_TYPE))
    edge_1 = [int(i.split()[1])-1 for i in bond_lines]
    edge_2 = [int(i.split()[2])-1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]
    x = torch.tensor(atoms)
    edge_idx=torch.tensor([edge_1+edge_2,edge_2+edge_1])
    edge_attr=torch.tensor(edge_attr+edge_attr)
    graph = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return graph


def mol2_to_graph_coords(mol2_file, ATOM_TYPE, using_hydrogen, input_type):
    '''
        Convert an Mol2 file to a list of lists of blocks for each molecule / residue.

        Parameters:
            mol2_file: Path to the Mol2 file
            using_hydrogen: Whether to preserve hydrogen atoms, default false
            input_type: "protein" or "ligand" (small molecule). If not specified, deduce from the mol2 file

        Returns:
            A list of blocks reprensenting a small molecule / protein, etc.
    '''
    # Read Mol2 file
    with open(mol2_file, 'r') as fin:
        lines = fin.readlines()

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n')+1:lines.index('@<TRIPOS>BOND\n')]
    if input_type == 'ligand':
        bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]
    else:
        bond_lines = lines[lines.index('@<TRIPOS>BOND\n')+1:lines.index('@<TRIPOS>SUBSTRUCTURE\n')]

    def line_to_atom(line):
        _, name, x, y, z, element, res_id, res_name, _ = re.split(r'\s+', line)[:9]
        element = element.split('.')[0]
        atom = Atom(name, [float(x), float(y), float(z)], element)
        return atom, res_id, res_name

    def extract_coord(line):
        indx = line.split()[0]
        element = line.split()[5].split('.')[0]
        x, y, z= line.split()[2:5]

        return indx, element, [float(x), float(y), float(z)]

    # to graph
    coord_list, atom_list, index_list = [], [], []
    cnt = 0
    for line in atom_lines:
        if len(line.split()) < 6:
            print('atom error!')
            print(line)
            continue
        indx, element, coord = extract_coord(line)
        if not using_hydrogen and element == 'H':
            continue
        coord_list.append(coord)
        atom_list.append(ATOM_TYPE.index(element)
                     if element in ATOM_TYPE
                     else len(ATOM_TYPE))
        index_list.append(indx)
    src_edge, dst_edge, edge_attr = [], [], []
    for line in bond_lines:
        _, src, dst, _type = line.split()
        if _type.isdigit():
            _type = int(_type)
        elif _type == 'ar':  # aromatic
            _type = 4
        elif _type == 'am':  # amide
            _type = 5
        elif _type in ['du', 'un', 'nc']:
            continue
        else:
            raise ValueError(f'bond type {_type} not recognized!')
        if src not in index_list or dst not in index_list:
            continue

        src, dst =  index_list.index(src), index_list.index(dst)
        src_edge.append(src)
        dst_edge.append(dst)
        edge_attr.append(_type)


    x = torch.tensor(atom_list)
    edge_idx = torch.tensor([src_edge + dst_edge, dst_edge + src_edge], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr + edge_attr)
    coords = torch.tensor(coord_list)
    graph = Data(x=x, coord=coords, edge_index=edge_idx, edge_attr=edge_attr)

    for idx in src_edge + dst_edge:
        if idx >= len(coord_list):
            raise ValueError(f'atom {idx} in {mol2_file} out of range!')

    return graph

def pdb_to_list_blocks(pdb: str, selected_chains=None):
    '''
        Convert pdb file to a list of lists of blocks using Biopython.
        Each chain will be a list of blocks.

        Parameters:
            pdb: Path to the pdb file
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']

        Returns:
            A list of lists of blocks. Each chain in the pdb file will be parsed into
            one list of blocks.
            example:
                [
                    [residueA1, residueA2, ...],  # chain A
                    [residueB1, residueB2, ...]   # chain B
                ],
                where each residue is instantiated by Block data class.
    '''

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('anonym', pdb)

    list_blocks, chain_ids = [], {}

    for chain in structure.get_chains():

        _id = chain.get_id()
        if (selected_chains is not None) and (_id not in selected_chains):
            continue

        residues, res_ids = [], {}
        for residue in chain:
            abrv = residue.get_resname()
            hetero_flag, res_number, insert_code = residue.get_id()
            res_id = f'{res_number}-{insert_code}'
            if hetero_flag == 'W':
                continue   # residue from glucose (WAT) or water (HOH)
            if hetero_flag.strip() != '' and res_id in res_ids:
                continue  # the solution (e.g. H_EDO (EDO))
            if abrv == 'MSE':
                abrv = 'MET'  # MET is usually transformed to MSE for structural analysis
            symbol = VOCAB.abrv_to_symbol(abrv)

            # filter Hs because not all data include them
            atoms = [ Atom(atom.get_id(), atom.get_coord(), atom.element) for atom in residue if atom.element != 'H' ]
            residues.append(Block(symbol, atoms))
            res_ids[res_id] = True

        # the last few residues might be non-relevant molecules in the solvent if their types are unk
        end = len(residues) - 1
        while end >= 0:
            if residues[end].symbol == VOCAB.UNK:
                end -= 1
            else:
                break
        residues = residues[:end + 1]
        if len(residues) == 0:  # not a chain
            continue

        chain_ids[_id] = len(list_blocks)
        # list_blocks.append(residues)
        list_blocks.extend(residues)

    return list_blocks


def pdb2graph(pdb_path):
    list_blocks = pdb_to_list_blocks(pdb_path)

    res = blocks_to_data(list_blocks)

    lengths = [len(res['B'])]
    res['lengths'] = torch.tensor(lengths, dtype=torch.long)

    keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
    types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]

    for key, _type in zip(keys, types):
        res[key] = torch.tensor(res[key], dtype=_type)

    Z, B, A, atom_positions, block_lengths, lengths, segment_ids = (
        res['X'], res['B'], res['A'], res['atom_positions'], res['block_lengths'], res['lengths'], res['segment_ids'])

    batch_id = torch.zeros_like(segment_ids)  # [Nb]
    batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
    batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

    block_id = torch.zeros_like(A)  # [Nu]
    block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
    block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

    args = get_args()
    edges, edge_attr = get_edges(B, batch_id, segment_ids, Z, block_id, args.k_neighbors, global_message_passing=False)

    block_embedding = BlockEmbedding(
        num_block_type=len(VOCAB),
        num_atom_type=VOCAB.get_num_atom_type(),
        num_atom_position=VOCAB.get_num_atom_pos(),
        embed_size=args.hidden_size,
        no_block_embedding=args.no_block_embedding)
    # x = block_embedding.block_embedding(B)
    x = B

    graph = Data(x=x, edge_index=edges, edge_attr=edge_attr)

    return graph


def get_edges(B, batch_id, segment_ids, Z, block_id, k_neighbors, global_message_passing):
    global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)
    edge_constructor = KNNBatchEdgeConstructor(
        k_neighbors=k_neighbors,
        global_message_passing=global_message_passing,
        global_node_id_vocab=[global_block_id],
        delete_self_loop=False)

    intra_edges, inter_edges, global_global_edges, global_normal_edges = construct_edges(
                edge_constructor, B, batch_id, segment_ids, Z, block_id, complexity=2000**2)
    if global_message_passing:
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        edge_attr = torch.cat([
            torch.zeros_like(intra_edges[0]),
            torch.ones_like(inter_edges[0])])
    else:
        edges = intra_edges
        edge_attr = torch.ones_like(intra_edges[0])
    # edge_embedding = nn.Embedding(4, 64)
    # edge_attr = edge_embedding(edge_attr)

    return edges, edge_attr

def pdb_to_mol2(pdb_path):

    mol2path = os.path.splitext(pdb_path)[0] + ".mol2"
    cmd.load(pdb_path, format='pdb')
    cmd.save(mol2path, format='mol2')
    cmd.delete('all')

    return mol2path

class GraphData(InMemoryDataset):
    def __init__(self, name, root="data", select_pocket_war=None, select_pocket_e3=None, conv_name='GCN'):
        self.select_pocket_war = select_pocket_war
        self.select_pocket_e3 = select_pocket_e3
        self.conv_name = conv_name

        super().__init__(root)

        if name == "protac":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif name == "ligase_pocket":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif name == "target_pocket":
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def processed_file_names(self):
        return ["protac.pt",
                "ligase_pocket.pt",
                "target_pocket.pt",
                'feature.pt',
                "label.pt",
                ]

    def process(self):
        with open(os.path.join(self.root, '{}.json'.format(args.dataset)), 'r') as f:
            name_dic = json.load(f)

        key_list = list(name_dic.keys())
        # key_list = []
        # for key, value in name_dic.items():
        #     if value['label'] == 0 or value['label'] == 1:
        #         key_list.append(key)

        print('Pocket selection starts!')
        tar_path_d, e3_path_d = {}, {}
        for key in key_list:
            tar_path = os.path.join(self.root, 'target_pocket', name_dic[key]['tar_path'])
            war_path = os.path.join(self.root, 'target_ligand', name_dic[key]['war_path'])
            e3_ligase_path = os.path.join(self.root, 'ligase_pocket', name_dic[key]['e3_ligase_path'])
            e3_lig_path = os.path.join(self.root, 'ligase_ligand', name_dic[key]['e3_lig_path'])

            if isinstance(self.select_pocket_war, int):
                print('Selection begin!')
                tar_name = basename(splitext(tar_path)[0])
                war_name = basename(splitext(war_path)[0])
                selection_name = '%s_pocket_%d' % (tar_name, self.select_pocket_war)
                tar_path_s = './data/PROTAC/selected_target/' + selection_name + '.mol2'
                if not isfile(tar_path_s):
                    print("Selecting residues within %d of warhead" % self.select_pocket_war)
                    cmd.load(tar_path, format='pdb')
                    cmd.load(war_path, format='mol2')
                    cmd.select(selection_name,
                               selection='(not %s) & br. all within %d of %s'
                                         % (war_name, self.select_pocket_war, war_name))
                    cmd.save(tar_path_s, selection_name)
                    cmd.delete('all')
                tar_path = tar_path_s
                tar_path_d[key] = tar_path
                # cmd.quit()
            if isinstance(self.select_pocket_e3, int):
                e3_name = basename(splitext(e3_ligase_path)[0])
                e3_lig_name = basename(splitext(e3_lig_path)[0])
                selection_name = '%s_pocket_%d' % (e3_name, self.select_pocket_e3)
                e3_ligase_path_s = './data/PROTAC/selected_e3/' + selection_name + '.mol2'
                if not isfile(e3_ligase_path_s):
                    print("Selecting residues within %d of e3 ligand" % self.select_pocket_e3)
                    cmd.load(e3_ligase_path, format='pdb')
                    cmd.load(e3_lig_path, format='mol2')

                    cmd.select(name=selection_name,
                               selection='(not %s) & br. all within %d of %s'
                                         % (e3_lig_name, self.select_pocket_e3, e3_lig_name))
                    cmd.save(e3_ligase_path_s, selection_name)
                    cmd.delete('all')
                e3_ligase_path = e3_ligase_path_s
                e3_path_d[key] = e3_ligase_path
        print('Pocket selection finished!')

        protac_graphs = []
        for key in key_list:
            if self.conv_name == 'EGNN':
                graph = mol2_to_graph_coords(os.path.join(self.root, 'protac', name_dic[key]['protac_path']), ATOMS,
                                             False, 'ligand')
            else:
                graph = mol2graph(os.path.join(self.root, 'protac', name_dic[key]['protac_path']), ATOMS, 'ligand')
            protac_graphs.append(graph)
        data, slices = self.collate(protac_graphs)
        torch.save((data, slices), self.processed_paths[0])
        print('protac graph processed!')

        ligase_pocket = []
        for key in key_list:
            if isinstance(self.select_pocket_e3, int):
                if self.conv_name == 'EGNN':
                    graph = mol2_to_graph_coords(e3_path_d[key], ATOMS, False, 'protein')
                else:
                    graph = mol2graph(e3_path_d[key], ATOMS, 'pocket')
            else:
                mol2path = pdb_to_mol2(os.path.join(self.root, 'ligase_pocket', name_dic[key]['e3_ligase_path']))
                if self.conv_name == 'EGNN':
                    graph = mol2_to_graph_coords(mol2path, ATOMS, False, 'protein')
                else:
                    graph = mol2graph(mol2path, ATOMS, 'protein')
                # graph = pdb2graph(os.path.join(self.root, 'ligase_pocket', name_dic[key]['e3_ligase_path']))
            ligase_pocket.append(graph)
        data, slices = self.collate(ligase_pocket)
        torch.save((data, slices), self.processed_paths[1])
        print('ligase_pocket processed!')

        target_pocket = []
        for key in key_list:
            try:
                if isinstance(self.select_pocket_war, int):
                    if self.conv_name == 'EGNN':
                        graph = mol2_to_graph_coords(tar_path_d[key], ATOMS, False, 'protein')
                    else:
                        graph = mol2graph(tar_path_d[key], ATOMS, 'pocket')
                else:
                    mol2path = pdb_to_mol2(os.path.join(self.root, 'target_pocket', name_dic[key]['tar_path']))
                    if self.conv_name == 'EGNN':
                        graph = mol2_to_graph_coords(mol2path, ATOMS, False, 'protein')
                    else:
                        graph = mol2graph(mol2path, ATOMS, 'protein')
                    # graph = pdb2graph(os.path.join(self.root, 'target_pocket', name_dic[key]['tar_path']))
            except:
                print('Something wrong!')
                print('{}: {}'.format(key, name_dic[key]))
            target_pocket.append(graph)
        data, slices = self.collate(target_pocket)
        torch.save((data, slices), self.processed_paths[2])
        print('target_pocket processed!')

        features = []
        with open('./data/PROTAC/features/protac_feature.npy', 'rb') as f:
            protac_feature = np.load(f, allow_pickle=True)

        with open('./data/PROTAC/features/target_feature.npy', 'rb') as f:
            target_feature = np.load(f, allow_pickle=True)

        with open('./data/PROTAC/features/e3_feature.npy', 'rb') as f:
            e3_feature = np.load(f, allow_pickle=True)

        feature = np.concatenate((protac_feature, target_feature, e3_feature), axis=1)
        torch.save(feature, self.processed_paths[3])
        print('Features processed!')

        labels = []
        for key in key_list:
            labels.append(name_dic[key]['label'])
        torch.save(labels, self.processed_paths[4])
        print('Labels processed!')

def construct_test_json(path, json_name):
    f = pd.read_csv(path, header=0, encoding='utf-8')
    if args.show_input:
        print(f.to_string(index=False))
    test_list = f[['Compound ID', 'Target', 'E3 ligase', 'Warhead', 'E3 ligand', 'Smiles']].values.tolist()

    case_dict = {}
    cnt = 0
    for item in test_list:
        comp_id, target, e3_ligase, warhead, e3_lig, smiles= item
        protac_path = smile2mol(smiles, comp_id)
        case_dict['c'+str(cnt)] = {"pro_comp_id": comp_id,
                                   "tar_path": f"{target}.pdb",
                                   'e3_ligase_path': f"{e3_ligase}.pdb",
                                   'war_path': f"{warhead}.mol2",
                                   'e3_lig_path': f"{e3_lig}.mol2",
                                   "protac_path": protac_path,
                                   'smiles': smiles,
                                   'label': -1}
        cnt += 1

    dir_path = os.path.dirname(path)
    with open(os.path.join(dir_path, f"{json_name}.json"), 'w') as f:
        json.dump(case_dict, f)

    return case_dict


if __name__=="__main__":
    ligase_ligand = GraphData("ligase_ligand")
