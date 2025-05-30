import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict

from .constants import (AA, max_num_heavyatoms,
                        restype_to_heavyatom_names, 
                        BBHeavyAtom,
                        ALL_ATOM_POSNS, ALL_ATOMS)
from .icoord import get_chi_angles, get_backbone_torsions
max_num_heavyatoms = 15
max_num_allatoms = 37

def _get_residue_heavyatom_info(res: Residue):
    # 获取ALL_ATOM_POSNS的info
    pos_allatom = torch.zeros([max_num_allatoms, 3], dtype=torch.float)
    mask_allatom = torch.zeros([max_num_allatoms, ], dtype=torch.bool)
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)

    restype = AA(res.get_resname())
    for atom_name in res:
        # if atom_name == '': continue
        if  atom_name.id not in ALL_ATOMS:
            # print("exist none-standard atoms.")
            continue
        # try:
        #     idx_heavyatom = restype_to_heavyatom_names[restype].index(atom_name.id)
        # except:
        #     continue
        idx_heavyatom = restype_to_heavyatom_names[restype].index(atom_name.id)

        idx_allatom = ALL_ATOM_POSNS[atom_name.id]
        pos_heavyatom[idx_heavyatom] = torch.tensor(res[atom_name.id].get_coord().tolist(), dtype=pos_heavyatom.dtype)
        mask_heavyatom[idx_heavyatom] = True
        pos_allatom[idx_allatom] = torch.tensor(res[atom_name.id].get_coord().tolist(), dtype=pos_allatom.dtype)
        mask_allatom[idx_allatom] = True

    return pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom


def parse_pdb(path, model_id, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path, model_id, assembly_id=0, unknown_threshold=1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if '_pdbx_struct_assembly_gen.asym_id_list' not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)
    else:
        assemblies = [tuple(chains.split(',')) for chains in mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']]
        label_to_auth = {}
        for label_asym_id, auth_asym_id in zip(mmcif_dict['_atom_site.label_asym_id'], mmcif_dict['_atom_site.auth_asym_id']):
            label_to_auth[label_asym_id] = auth_asym_id
        model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
        return parse_biopython_structure(model_real)


def parse_biopython_structure(entity, chains_ordered,  unknown_threshold=1.0):
    chains = Selection.unfold_entities(entity, 'C')
    # delete invalid chain
    chains = [c for c in chains if c.id != " "]
    
    # Create mapping from chain ID to index in ordered list
    index_dict = {chain_id: i for i, chain_id in enumerate(chains_ordered)}
    
    # Sort chains by their chain IDs using the index mapping
    chains = sorted(chains, key=lambda x: index_dict[x.id])

    # Initialize lists to store data from all chains
    all_chain_nb = []
    all_resseq = []
    all_res_nb = []
    all_residue_idx = []
    all_aa = []
    all_pos_heavyatom = []
    all_mask_heavyatom = []
    all_pos_allatom = []
    all_mask_allatom = []
    all_phi = []
    all_phi_mask = []
    all_psi = []
    all_psi_mask = []
    all_chi = []
    all_chi_alt = []
    all_chi_mask = []
    all_chi_complete = []

    count_unk = 0
    c = 1
    l0 = 0
    chain_id = 0
    for i, chain in enumerate(chains):
        if chain.get_id() == ' ':
            continue
        count_aa = 0
        data = EasyDict({
            'chain_nb': [],
            'resseq': [], 'res_nb': [], 'residue_idx': [],
            'aa': [],
            'pos_heavyatom': [], 'mask_heavyatom': [],
            'pos_allatom': [], 'mask_allatom': [],
            'phi': [], 'phi_mask': [],
            'psi': [], 'psi_mask': [],
            'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
        })

        chain.atom_to_internal_coordinates()
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_nb.append(chain_id)

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.pos_allatom.append(pos_allatom)
            data.mask_allatom.append(mask_allatom)

            # Backbone torsions
            phi, psi, _ = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.res_nb.append(seq_this)
            data.residue_idx.append(100 * (c - 1) + l0 + seq_this)

        l0 += seq_this
        c += 1
        if len(data.aa) == 0:
            continue

        # Append data from this chain to the lists
        all_chain_nb.extend(data.chain_nb)
        all_resseq.extend(data.resseq)
        all_res_nb.extend(data.res_nb)
        all_residue_idx.extend(data.residue_idx)
        all_aa.extend(data.aa)
        all_pos_heavyatom.extend(data.pos_heavyatom)
        all_mask_heavyatom.extend(data.mask_heavyatom)
        all_pos_allatom.extend(data.pos_allatom)
        all_mask_allatom.extend(data.mask_allatom)
        all_phi.extend(data.phi)
        all_phi_mask.extend(data.phi_mask)
        all_psi.extend(data.psi)
        all_psi_mask.extend(data.psi_mask)
        all_chi.extend(data.chi)
        all_chi_alt.extend(data.chi_alt)
        all_chi_mask.extend(data.chi_mask)
        all_chi_complete.extend(data.chi_complete)
        chain_id += 1

    if len(all_aa) == 0:
        return None, None

    if (count_unk / l0) >= unknown_threshold:
        return None, None

    # Convert lists to tensors
    data_dict = {
        'chain_nb': torch.tensor(all_chain_nb, dtype=torch.long),
        'resseq': torch.tensor(all_resseq, dtype=torch.long),
        'res_nb': torch.tensor(all_res_nb, dtype=torch.long),
        'residue_idx': torch.tensor(all_residue_idx, dtype=torch.long),
        'aa': torch.tensor(all_aa, dtype=torch.long),
        'pos_heavyatom': torch.stack(all_pos_heavyatom),
        'mask_heavyatom': torch.stack(all_mask_heavyatom),
        'pos_allatom': torch.stack(all_pos_allatom),
        'mask_allatom': torch.stack(all_mask_allatom),
        'phi': torch.tensor(all_phi, dtype=torch.float),
        'phi_mask': torch.tensor(all_phi_mask, dtype=torch.bool),
        'psi': torch.tensor(all_psi, dtype=torch.float),
        'psi_mask': torch.tensor(all_psi_mask, dtype=torch.bool),
        'chi': torch.stack(all_chi),
        'chi_alt': torch.stack(all_chi_alt),
        'chi_mask': torch.stack(all_chi_mask),
        'chi_complete': torch.tensor(all_chi_complete, dtype=torch.bool)
    }

    return data_dict, l0
