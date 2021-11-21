import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import RDConfig, DataStructs
from rdkit.Chem import HybridizationType, ChemicalFeatures, rdDepictor, MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import zscore
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer
from torch import load, Tensor, long, zeros, manual_seed
from random import seed
import shap
from .architecture import EGConvNet
from standardiser import standardise

fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))

regression_tasks = ['Caco2_Wang', 'Lipophilicity_AstraZeneca','Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo',
                     'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'LD50_Zhu']
classification_tasks = ['HIA_Hou','Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins', 'CYP2C19_Veith',
                    'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith','CYP2C9_Substrate_CarbonMangels',
                     'CYP2D6_Substrate_CarbonMangels','CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI',
                    'Skin Reaction', 'Carcinogens_Languin','ClinTox','nr-ar', 'nr-ar-lbd', 'nr-ahr', 'nr-aromatase',
                     'nr-er','nr-er-lbd', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp','sr-p53']

root = root = Path(__file__).resolve().parents[0].absolute()

def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class DAO:
    def __init__(self):
        state_dict = load(root / 'state_dict.pt')
        hyperparams = state_dict['hyper_parameters']
        self.model = EGConvNet(
            hyperparams['hidden_channels'],
            hyperparams['num_layers'],
            hyperparams['num_heads'],
            hyperparams['num_bases'],
            aggregator=['sum', 'mean', 'max'])
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.eval()

    def smiles2graph(self, smiles):
        """
        Converts SMILES string to graph Data object
        :input: SMILES string (str)
        :return: graph object
        """

        smiles = standardise.run(r'{}'.format(smiles))
        mol = MolFromSmiles(r'{}'.format(smiles))

        # atoms
        donor = []
        acceptor = []
        features = []
        names = []
        donor_string = []

        for atom in mol.GetAtoms():
            atom_feature_names = []
            atom_features = []
            atom_features += one_hot_vector(
                atom.GetAtomicNum(),
                [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
            )

            atom_feature_names.append(atom.GetSymbol())
            atom_features += one_hot_vector(
                atom.GetTotalNumHs(),
                [0, 1, 2, 3, 4]
            )
            atom_feature_names.append(atom.GetTotalNumHs())
            atom_features += one_hot_vector(
                atom.GetHybridization(),
                [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
                 HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED]
            )
            atom_feature_names.append(atom.GetHybridization().__str__())

            atom_features.append(atom.IsInRing())
            atom_features.append(atom.GetIsAromatic())

            if atom.GetIsAromatic() == 1:
                atom_feature_names.append('Aromatic')
            else:
                atom_feature_names.append('Non-aromatic')

            if atom.IsInRing() == 1:
                atom_feature_names.append('Is in ring')
            else:
                atom_feature_names.append('Not in ring')

            donor.append(0)
            acceptor.append(0)

            donor_string.append('Not a donor or acceptor')

            atom_features = np.array(atom_features, dtype=int)
            atom_feature_names = np.array(atom_feature_names, dtype=object)
            features.append(atom_features)
            names.append(atom_feature_names)

        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 0
                    donor_string[k] = 'Donor'
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1
                    donor_string[k] = 'Acceptor'

        features = np.array(features, dtype=int)
        donor = np.array(donor, dtype=int)
        donor = donor[..., np.newaxis]
        acceptor = np.array(acceptor, dtype=int).transpose()
        acceptor = acceptor[..., np.newaxis]
        x = np.append(features, donor, axis=1)
        x = np.append(x, acceptor, axis=1)

        donor_string = np.array(donor_string, dtype=object)
        donor_string = donor_string[..., np.newaxis]

        names = np.array(names, dtype=object)
        names = np.append(names, donor_string, axis=1)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # add edges in both directions
                edges_list.append((i, j))
                edges_list.append((j, i))

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)

        graph = dict()
        graph['edge_index'] = Tensor(edge_index).long()
        graph['node_feat'] = Tensor(x)
        graph['feature_names'] = names

        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], feature_names=names)

    def predict(self, smiles):

        mol = MolFromSmiles(r'{}'.format(smiles))
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')

        # smiles = standardise.run(r'{}'.format(smiles))
        data = self.smiles2graph(r'{}'.format(smiles))
        data.batch = zeros(data.num_nodes, dtype=long)
        output = self.model(data.x, data.edge_index, data.batch).detach().cpu().numpy()[0][0]
        output = round(((1 / (1 + np.exp(-output))) * 100), 2)
        predicted_class = 1 if output > 53 else 0
        approved_calibration = np.loadtxt(root / 'approved_calibration.csv') * 100
        withdrawn_calibration = np.loadtxt(root / 'withdrawn_calibration.csv') * 100
        approved_p_value = (np.searchsorted(approved_calibration, (100 - output))) \
                           / (len(approved_calibration) + 1)
        withdrawn_p_value = (np.searchsorted(withdrawn_calibration, output)) \
                            / (len(withdrawn_calibration) + 1)


        return predicted_class, output, approved_p_value, withdrawn_p_value

    def explain(self, smiles):
        features = ["boron", "carbon", "nitrogen", "oxygen", "flourine", "phosporus", "sulfur",
                    "chlorine", "bromine", "iodine", "other", "zero_Hs", "one_H", "two_Hs", "three_Hs",
                    "four_Hs", "s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "unspecified_hybr",
                    "in_ring", "aromatic", "donor", "acceptor"]

        # set seeds for initializing mask in GNN explainer
        manual_seed(0)
        seed(0)
        np.random.seed(0)

        explainer = GNNExplainer(self.model, epochs=100)
        smiles = standardise.run(r'{}'.format(smiles))
        data = self.smiles2graph(smiles)
        data.batch = zeros(data.num_nodes, dtype=long)
        node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)

        # node importance
        node_feat_mask = node_feat_mask.detach().numpy()

        # edge importance
        edge_mask = edge_mask.detach().numpy()
        edge_mask = abs(zscore(edge_mask))
        highlighted_edges = list((np.where(edge_mask >= 1)[0]).astype(object))
        edge_index = data.edge_index.detach().cpu().numpy()

        # edge indices contain both direction so we need to drop one and save a copy
        final_edge = edge_index[:, ::2]
        normal = []  # first direction in edge mask
        reverse = []  # second direction
        for high in highlighted_edges:
            normal.append(list(edge_index[:, high]))
            reverse.append(list(edge_index[:, high][::-1]))

        # find bonds to highlight
        bonds_to_highlight = []
        for i in range(len(final_edge[0])):
            atom_1 = final_edge[0][i]
            atom_2 = final_edge[1][i]
            bond = [atom_1, atom_2]
            if bond in normal:
                bonds_to_highlight.append(i)
                continue
            if bond in reverse:
                bonds_to_highlight.append(i)

        mol = MolFromSmiles(r'{}'.format(smiles))
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=bonds_to_highlight)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        molecule_vis = svg

        return molecule_vis, dict(zip(features, node_feat_mask))

    def complementary_model(self, smiles):
        compl_root = Path(__file__).resolve().parents[1].absolute()

        smiles = standardise.run(r'{}'.format(smiles))
        withdrawn_prob = self.predict(smiles)[1] #return the probability

        predictions = []
        predictions.append(round(withdrawn_prob / 100, 2))
        tasks = ['predict_withdrawn',
                 'CYP2C9_Substrate_CarbonMangels',
                 'nr-ppar-gamma',
                 'Bioavailability_Ma',
                 'Clearance_Hepatocyte_AZ']

        data = self.smiles2graph(r'{}'.format(smiles))
        data.batch = zeros(data.num_nodes, dtype=long)

        for task in tasks[1:]:
            state_dict = load(compl_root / 'complementary/{}/state_dict.pt'.format(task))
            hyperparams = state_dict['hyper_parameters']
            compl_model = EGConvNet(
                hyperparams['hidden_channels'],
                hyperparams['num_layers'],
                hyperparams['num_heads'],
                hyperparams['num_bases'],
                aggregator=['sum', 'mean', 'max'])
            compl_model.load_state_dict(state_dict['state_dict'])
            compl_model.eval()

            output = compl_model(data.x, data.edge_index, data.batch).detach().cpu().numpy()[0][0]
            if task in classification_tasks:
                output = round(((1 / (1 + np.exp(-output)))), 2)
            else:
                output = round(output, 2)
            predictions.append(output.astype(float))


        test_example = pd.DataFrame(columns=tasks, data=[predictions], index=[0])

        xgb_file = open(compl_root / 'complementary/xgb_classifier_reduced.pkl', 'rb')
        xgb_model = pickle.load(xgb_file)
        ntree_limit = xgb_model.get_booster().best_ntree_limit

        prediction = int(xgb_model.predict(test_example, ntree_limit=ntree_limit)[0])
        probability = xgb_model.predict_proba(test_example, ntree_limit=ntree_limit)[0][1]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(test_example)

        test_example.rename(columns={'Clearance_Hepatocyte_AZ': 'Clearance Hepatocyte',
                                     'Bioavailability_Ma': 'Bioavailability',
                                     'CYP2C9_Substrate_CarbonMangels': 'CYP2C9 Substrate',
                                     'predict_withdrawn': 'Predict withdrawn'}, inplace=True)

        return {'predicted_class': prediction, 'withdrawn_probability': probability}, dict(zip(tasks, predictions)), dict(zip(tasks, shap_values.values.ravel()))

